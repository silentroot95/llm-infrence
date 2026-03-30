
#include <limits>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <stdexcept>

#include "cJSON.h"

#include "mmap.h"
#include "tokenizer.h"
#include "tensor.h"
#include "operator.h"



static inline uint64_t pack_key(int a, int b) {
    return ((uint64_t)a << 32) | (uint32_t)b;
}

static std::string ltrim_newline_copy(const std::string& s) {
    size_t start = 0;
    while (start < s.size() && (s[start] == '\n' || s[start] == '\r')) {
        start++;
    }
    return s.substr(start);
}

static std::string rtrim_newline_copy(const std::string& s) {
    size_t end = s.size();
    while (end > 0 && (s[end - 1] == '\n' || s[end - 1] == '\r')) {
        end--;
    }
    return s.substr(0, end);
}

static bool starts_with(const std::string& s, const std::string& prefix) {
    return s.size() >= prefix.size() && s.compare(0, prefix.size(), prefix) == 0;
}

static bool ends_with(const std::string& s, const std::string& suffix) {
    return s.size() >= suffix.size() &&
           s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

static std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (char c : s) {
        switch (c) {
            case '\\': out += "\\\\"; break;
            case '"': out += "\\\""; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default: out.push_back(c); break;
        }
    }
    return out;
}

static std::string strip_newline_copy(const std::string& s) {
    size_t start = 0;
    while (start < s.size() && (s[start] == '\n' || s[start] == '\r')) {
        start++;
    }

    size_t end = s.size();
    while (end > start && (s[end - 1] == '\n' || s[end - 1] == '\r')) {
        end--;
    }

    return s.substr(start, end - start);
}

static size_t find_next_added_token(
    const std::string& text,
    size_t start,
    const std::vector<std::string>& added_tokens,
    std::string& matched_token
) {
    size_t best_pos = std::string::npos;
    const std::string* best_token = nullptr;

    for (const std::string& token : added_tokens) {
        size_t pos = text.find(token, start);
        if (pos == std::string::npos) {
            continue;
        }

        if (best_pos == std::string::npos || pos < best_pos ||
            (pos == best_pos && best_token != nullptr && token.size() > best_token->size())) {
            best_pos = pos;
            best_token = &token;
        }
    }

    if (best_token != nullptr) {
        matched_token = *best_token;
    } else {
        matched_token.clear();
    }
    return best_pos;
}

std::vector<std::string> Tokenizer::byte_split(const std::string& text) {
    std::vector<std::string> tokens;
    for(uint8_t b : text) {
        tokens.push_back(encoder.byte_to_str[b]);
    }
    return tokens;
}

int Tokenizer::find_best_merge(const std::vector<std::string>& tokens, int& best_idx) {
    int best_rank = std::numeric_limits<int>::max();
    best_idx = -1;

    for (int i = 0; i < (int)tokens.size() - 1; i++) {
        auto it1 = vocab.find(tokens[i]);
        auto it2 = vocab.find(tokens[i + 1]);

        if (it1 == vocab.end() || it2 == vocab.end()) {
            continue; 
        }

        int id1 = it1->second;
        int id2 = it2->second;

        uint64_t key = pack_key(id1, id2);

        auto it = merges.find(key);
        if (it != merges.end() && it->second < best_rank) {
            best_rank = it->second;
            best_idx = i;
        }
    }
    return best_rank;
}

//===== BPE merge =====
std::vector<std::string> Tokenizer::bpe(const std::string& text) {
    std::vector<std::string> tokens = byte_split(text);

    while (true) {
        int best_idx;
        int best_rank = find_best_merge(tokens, best_idx);
        if (best_idx == -1) break;

        tokens[best_idx] = tokens[best_idx] + tokens[best_idx + 1];
        tokens.erase(tokens.begin() + best_idx + 1);
    }
    return tokens;
}

std::vector<int> Tokenizer::encode(const std::string& text) {
    std::vector<int> token_ids;

    auto append_bpe_tokens = [&](const std::string& chunk) {
        if (chunk.empty()) {
            return;
        }

        std::vector<std::string> tokens = bpe(chunk);
        for (const std::string& t : tokens) {
            auto it = vocab.find(t);
            if (it == vocab.end()) {
                fprintf(stderr, "Fatal error, unknown token %s\n", t.c_str());
                exit(EXIT_FAILURE);
            }
            token_ids.push_back(it->second);
            //data[ids.shape[0]++] = it->second;
        }
    };

    if (added_tokens.empty()) {
        append_bpe_tokens(text);
        return token_ids;
    }

    size_t cursor = 0;
    while (cursor < text.size()) {
        std::string matched_token;
        size_t token_pos = find_next_added_token(text, cursor, added_tokens, matched_token);

        if (token_pos == std::string::npos) {
            append_bpe_tokens(text.substr(cursor));
            break;
        }

        if (token_pos > cursor) {
            append_bpe_tokens(text.substr(cursor, token_pos - cursor));
        }

        auto it = vocab.find(matched_token);
        if (it == vocab.end()) {
            fprintf(stderr, "Fatal error, unknown added token %s\n", matched_token.c_str());
            exit(EXIT_FAILURE);
        }
        token_ids.push_back(it->second);
        // data[ids.shape[0]++] = it->second;
        cursor = token_pos + matched_token.size();
    }
    return token_ids;
}

std::string Tokenizer::apply_chat_template(
    const std::vector<ChatMessage>& messages,
    bool add_generation_prompt,
    const std::vector<ToolDefinition>& tools,
    bool enable_thinking
) const {
    if (messages.empty()) {
        throw std::runtime_error("No messages provided.");
    }

    std::string prompt;

    if (!tools.empty()) {
        prompt += "<|im_start|>system\n";
        if (messages[0].role == "system" && !messages[0].content.empty()) {
            prompt += messages[0].content;
            prompt += "\n\n";
        }

        prompt += "# Tools\n\nYou may call one or more functions to assist with the user query.\n\n";
        prompt += "You are provided with function signatures within <tools></tools> XML tags:\n<tools>";
        for (const ToolDefinition& tool : tools) {
            prompt += "\n";
            prompt += tool.json;
        }
        prompt += "\n</tools>\n\n";
        prompt += "For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n";
        prompt += "<tool_call>\n";
        prompt += "{\"name\": <function-name>, \"arguments\": <args-json-object>}\n";
        prompt += "</tool_call><|im_end|>\n";
    } else if (messages[0].role == "system") {
        prompt += "<|im_start|>system\n";
        prompt += messages[0].content;
        prompt += "<|im_end|>\n";
    }

    bool multi_step_tool = true;
    int last_query_index = (int)messages.size() - 1;
    for (int i = (int)messages.size() - 1; i >= 0; --i) {
        const ChatMessage& message = messages[i];
        if (multi_step_tool && message.role == "user") {
            if (!(starts_with(message.content, "<tool_response>") &&
                  ends_with(message.content, "</tool_response>"))) {
                multi_step_tool = false;
                last_query_index = i;
            }
        }
    }

    for (int i = 0; i < (int)messages.size(); ++i) {
        const ChatMessage& message = messages[i];
        std::string content = message.content;

        if ((message.role == "user") || (message.role == "system" && i != 0)) {
            prompt += "<|im_start|>";
            prompt += message.role;
            prompt += "\n";
            prompt += content;
            prompt += "<|im_end|>\n";
            continue;
        }

        if (message.role == "assistant") {
            std::string reasoning_content = message.reasoning_content;
            if (reasoning_content.empty()) {
                size_t think_end = content.find("</think>");
                if (think_end != std::string::npos) {
                    std::string before = rtrim_newline_copy(content.substr(0, think_end));
                    size_t think_start = before.rfind("<think>");
                    if (think_start != std::string::npos) {
                        reasoning_content = ltrim_newline_copy(before.substr(think_start + 7));
                        content = ltrim_newline_copy(content.substr(think_end + 8));
                    }
                }
            }

            prompt += "<|im_start|>assistant\n";
            if (i > last_query_index) {
                if (i == (int)messages.size() - 1 || !reasoning_content.empty()) {
                    prompt += "<think>\n";
                    prompt += strip_newline_copy(reasoning_content);
                    prompt += "\n</think>\n\n";
                    prompt += ltrim_newline_copy(content);
                } else {
                    prompt += content;
                }
            } else {
                prompt += content;
            }

            for (size_t j = 0; j < message.tool_calls.size(); ++j) {
                const ToolCall& tool_call = message.tool_calls[j];
                if ((j == 0 && !content.empty()) || j > 0) {
                    prompt += "\n";
                }
                prompt += "<tool_call>\n";
                prompt += "{\"name\": \"";
                prompt += json_escape(tool_call.name);
                prompt += "\", \"arguments\": ";
                prompt += tool_call.arguments_json.empty() ? "{}" : tool_call.arguments_json;
                prompt += "}\n</tool_call>";
            }
            prompt += "<|im_end|>\n";
            continue;
        }

        if (message.role == "tool") {
            if (i == 0 || messages[i - 1].role != "tool") {
                prompt += "<|im_start|>user";
            }

            prompt += "\n<tool_response>\n";
            prompt += content;
            prompt += "\n</tool_response>";

            if (i == (int)messages.size() - 1 || messages[i + 1].role != "tool") {
                prompt += "<|im_end|>\n";
            }
            continue;
        }
    }

    if (add_generation_prompt) {
        prompt += "<|im_start|>assistant\n";
        if (!enable_thinking) {
            prompt += "<think>\n\n</think>\n\n";
        }
    }

    return prompt;
}


void Tokenizer::load_vocab(const char* path) {
    size_t size;
    void* data = memory_map(path, &size);

    const char* p = (const char*)data;
    const char* end = p + size;

    std::string key;
    int value = 0;

    bool in_string = false;
    bool escape = false;
    bool reading_value = false;

    while (p < end) {
        char c = *p++;

        if (in_string) {
            if (escape) {
                // 处理转义字符
                switch (c) {
                    case '"': key.push_back('"'); break;
                    case '\\': key.push_back('\\'); break;
                    case 'n': key.push_back('\n'); break;
                    case 't': key.push_back('\t'); break;
                    default: key.push_back(c); break;
                }
                escape = false;
            }
            else if (c == '\\') {
                escape = true;
            }
            else if (c == '"') {
                in_string = false; 
            }
            else {
                key.push_back(c);
            }
        }
        else if (reading_value) {
            if (std::isdigit(c)) {
                value = value * 10 + (c - '0');
            }
            else {
                vocab[key] = value;
                id_to_token[value] = key;
                key.clear();
                value = 0;
                reading_value = false;
            }
        }
        else {
            if (c == '"') {
                in_string = true;
                key.clear();
            }
            else if (c == ':') {
                reading_value = true;
                value = 0;
            }
        }
    }

    if (reading_value && !key.empty()) {
        vocab[key] = value;
        id_to_token[value] = key;
    }
    memory_unmap(data, size);
}


void Tokenizer::load_merges(const char* path) {
    size_t size;
    void* data = memory_map(path, &size);

    const char* p = (const char*)data;
    const char* end = p + size;

    int rank = 0;

    while (p < end) {
        // ===== 1. 读取一行 =====
        const char* line_start = p;
        while (p < end && *p != '\n') p++;
        const char* line_end = p;
        if (p < end) p++; // 跳过 '\n'

        // 跳过空行
        if (line_start == line_end) continue;

        // 跳过注释
        //if (*line_start == '#') continue;

        // ===== 2. split：找空格 =====
        const char* space = line_start;
        while (space < line_end && *space != ' ') space++;
        if (space == line_end) continue; // 格式异常

        // token A
        std::string a(line_start, space - line_start);

        // token B（注意处理 \r）
        const char* b_start = space + 1;
        const char* b_end = line_end;

        if (b_end > b_start && *(b_end - 1) == '\r') {
            b_end--; // 去掉 Windows \r
        }

        std::string b(b_start, b_end - b_start);

        // ===== 3. vocab 查 id =====
        auto it1 = vocab.find(a);
        auto it2 = vocab.find(b);

        if (it1 == vocab.end() || it2 == vocab.end()) {
            continue; // tokenizer 允许跳过异常
        }

        int id1 = it1->second;
        int id2 = it2->second;

        // ===== 4. pack + 存 rank =====
        uint64_t key = pack_key(id1, id2);
        merges[key] = rank++;
    }

    memory_unmap(data, size);
}

std::string Tokenizer::decode(int id) {
    std::string tokens = id_to_token[id];
    std::string decode_token;
    for(auto iter=tokens.begin();iter<tokens.end();) {
        uint8_t c = (uint8_t)*iter;
        size_t len = 0;

        if((c & 0x80) == 0x00) {
            len = 1;
        }
        else if ((c & 0xE0) == 0xC0) {
            len = 2;
        }
        else if ((c & 0xF0) == 0xE0) {
            len = 3;
        }
        else if ((c& 0xF8) == 0xF0) {
            len = 4;
        }
        else {
            throw std::runtime_error("invalid UTF-8");
        }

        if (iter + len > tokens.end()) {
                throw std::runtime_error("truncated UTF-8");
        }
        std::string key(iter,iter+len);

        auto it = encoder.str_to_byte.find(key);
        if (it == encoder.str_to_byte.end()) {
            throw std::runtime_error("unknown unicode char");
        }

        decode_token.push_back((char)it->second);
        iter += len;
    }
    return decode_token;
}

void Tokenizer::load_tokenizer_config(const char* path) {
    size_t size;
    void* data = memory_map(path, &size);
    char* json_string = (char*)malloc(size + 1);
    memcpy(json_string, data, size);
    json_string[size] = '\0';

    memory_unmap(data, size);

    cJSON* json = cJSON_Parse(json_string);
    if (json == NULL) {
        fprintf(stderr, "Error parsing JSON: %s\n", cJSON_GetErrorPtr());
        free(json_string);
        exit(EXIT_FAILURE);
    }

    cJSON* object = cJSON_GetObjectItemCaseSensitive(json, "added_tokens_decoder");
    cJSON* element = NULL;
    cJSON_ArrayForEach(element, object) {
        int tokenid = atoi(element->string);
        const char* token_str = cJSON_GetObjectItemCaseSensitive(element, "content")->valuestring;

        if (token_str) {
            std::string token(token_str);
            vocab[token] = tokenid;
            id_to_token[tokenid] = token;
            added_tokens.push_back(token);
        }
    }

    std::sort(added_tokens.begin(), added_tokens.end(),
        [](const std::string& a, const std::string& b) {
            if (a.size() != b.size()) return a.size() > b.size();
            return a < b;
        });
    added_tokens.erase(std::unique(added_tokens.begin(), added_tokens.end()), added_tokens.end());

    free(json_string);
    cJSON_Delete(json);
}


void Tokenizer::load_gen_config(const char* path) {
    size_t size = 0;
    void* data = memory_map(path, &size);

    char* json_string = (char*)malloc(size + 1);
    memcpy(json_string,data,size);
    json_string[size] = '\0';

    memory_unmap(data, size);

    cJSON* json = cJSON_Parse(json_string);
    if (json==NULL){
        fprintf(stderr,"Error parsing JSON: %s\n", cJSON_GetErrorPtr());
        free(json_string);
        exit(EXIT_FAILURE);
    }
    cJSON* item = cJSON_GetObjectItemCaseSensitive(json,"bos_token_id");
    bos_token_id = item->valueint;

    item = cJSON_GetObjectItemCaseSensitive(json, "eos_token_id");

    cJSON* element = NULL;
    if(cJSON_IsNumber(item)) {
        eos_token_id.push_back(item->valueint);
    }
    else if(cJSON_IsArray(item)) {
        cJSON_ArrayForEach(element, item) {
            eos_token_id.push_back(element->valueint);
        }
    }

    item = cJSON_GetObjectItemCaseSensitive(json, "pad_token_id");
    pad_token_id = item->valueint;

    item = cJSON_GetObjectItemCaseSensitive(json, "temperature");
    temperature = item->valuedouble;
 
    item = cJSON_GetObjectItemCaseSensitive(json, "top_p");
    top_p = item->valuedouble;

    item = cJSON_GetObjectItemCaseSensitive(json, "top_k");
    top_k = item->valueint;

    free(json_string);
    cJSON_Delete(json);
}

Sampler::Sampler(int vocab_size,float t,float p,int k) : 
    m_temperature(t),
    m_top_k(k),
    m_top_p(p) {
        m_indices.resize(vocab_size);
        for (int i = 0; i < vocab_size; i++) {
            m_indices[i] = i;
        }
    }

int Sampler::sample(const Tensor* data, bool do_sample) {

    if(!do_sample) {
        return argmax(data);
    }

    float* logits = (float*) data->data;
    int vocab_size = data->shape[1];

    //temperature softmax
    softmax(logits, vocab_size, m_temperature);

    std::vector<float> probs(logits, logits + vocab_size);
    // top-k 
    int k = m_top_k > 0 ? std::min(m_top_k, vocab_size) : vocab_size;

    if (k < vocab_size) {
        std::partial_sort(
            m_indices.begin(),
            m_indices.begin() + k,
            m_indices.end(),
            [&](int a, int b) {
                return probs[a] > probs[b];
            }
        );
    }

    // top-p 
    int cutoff = k;
    if (m_top_p < 1.0f) {
        float cumulative = 0.0f;
        for (int i = 0; i < k; i++) {
            cumulative += probs[m_indices[i]];
            if (cumulative >= m_top_p) {
                cutoff = i + 1;
                break;
            }
        }
    }

    std::vector<int> final_idx;
    std::vector<float> final_probs;

    float final_sum = 0.0f;
    for (int i = 0; i < cutoff; i++) {
        int id = m_indices[i];
        final_idx.push_back(id);
        final_probs.push_back(probs[id]);
        final_sum += probs[id];
    }

    for (float& p : final_probs) {
        p /= final_sum;
    }

    return sample_multinomial(final_idx, final_probs);
}

int Sampler::sample_multinomial(const std::vector<int>& idx,
                                const std::vector<float>& p) {

    float r = (float)rand() / RAND_MAX;

    float cumulative = 0.0f;
    for (size_t i = 0; i < p.size(); i++) {
        cumulative += p[i];
        if (r <= cumulative) {
            return idx[i];
        }
    }

    return idx.back();
}

