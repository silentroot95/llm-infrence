
#include <limits>
#include <algorithm>
#include <cmath>

#include "cJSON.h"

#include "mmap.h"
#include "tokenizer.h"
#include "tensor.h"



static inline uint64_t pack_key(int a, int b) {
    return ((uint64_t)a << 32) | (uint32_t)b;
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

void Tokenizer::encode(const std::string& text, Tensor& ids) {
    std::vector<std::string> tokens = bpe(text);
    ids.shape[0] = 0;
    int* data = (int*)ids.data;

    for (auto& t : tokens) {
        if (vocab.count(t)) {
            data[ids.shape[0]] = vocab[t];
        } else {
            fprintf(stderr, "Fatal error, unknown token %s\n",t);
            //data[ids.shape[0]] = -1;
            exit(EXIT_FAILURE);
        }
        ids.shape[0]++;
    }
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



Sampler::Sampler(int vocab_size,float t,int k,float p) : 
    temperature(t),
    top_k(k),
    top_p(p) {
        probs.resize(vocab_size);
        indices.resize(vocab_size);
    }

int Sampler::sample(Tensor* data) {
    
    float* logits = (float*) data->data;
    int vocab_size = data->shape[1];

    //temperature + softmax
    float max_logit = logits[0];
    for (int i = 1; i < vocab_size; i++) {
        if (logits[i] > max_logit) max_logit = logits[i];
    }

    float sum = 0.0f;
    for (int i = 0; i < vocab_size; i++) {
        float val = (logits[i] - max_logit) / temperature;
        probs[i] = expf(val);
        sum += probs[i];
        indices[i] = i;
    }

    for (int i = 0; i < vocab_size; i++) {
        probs[i] /= sum;
    }

    // top-k 
    int k = top_k > 0 ? std::min(top_k, vocab_size) : vocab_size;

    if (k < vocab_size) {
        std::partial_sort(
            indices.begin(),
            indices.begin() + k,
            indices.end(),
            [&](int a, int b) {
                return probs[a] > probs[b];
            }
        );
    }

    // top-p 
    int cutoff = k;
    if (top_p < 1.0f) {
        float cumulative = 0.0f;
        for (int i = 0; i < k; i++) {
            cumulative += probs[indices[i]];
            if (cumulative >= top_p) {
                cutoff = i + 1;
                break;
            }
        }
    }

    std::vector<int> final_idx;
    std::vector<float> final_probs;

    float final_sum = 0.0f;
    for (int i = 0; i < cutoff; i++) {
        int id = indices[i];
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