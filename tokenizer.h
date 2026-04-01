#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <stdint.h>


struct Tensor;

struct ByteEncoder {
    std::unordered_map<uint8_t, std::string> byte_to_str;
    std::unordered_map<std::string, uint8_t> str_to_byte;

    ByteEncoder() {
        int n = 0;
        for(int i=0; i<256; ++i) {
            uint32_t unicode_cp;
            if( (i>=33 && i<=126)   ||
                (i>=161 && i<=172)  ||
                (i>=174 && i<=255) ) {
                    unicode_cp = i;
            }
            else{
                unicode_cp = 256+n;
                ++n;
            }
            std::string s;
            if (unicode_cp < 0x80) {
                s.push_back((char)unicode_cp);
            }
            else if (unicode_cp < 0x800) {
                s.push_back(0xC0 | (unicode_cp >> 6));
                s.push_back(0x80 | (unicode_cp & 0x3F));
            }
            else {
                s.push_back(0xE0 | (unicode_cp >> 12));
                s.push_back(0x80 | ((unicode_cp >> 6) & 0x3F));
                s.push_back(0x80 | (unicode_cp & 0x3F));
            }
            byte_to_str[(uint8_t)i] = s;
            str_to_byte[s] = (uint8_t)i;
        }
    }
};

struct ToolCall {
    std::string name;
    std::string arguments_json;
};

struct ToolDefinition {
    std::string json;
};

struct ChatMessage {
    std::string role;
    std::string content;
    std::string reasoning_content;
    std::vector<ToolCall> tool_calls;
};

struct Tokenizer {
    //using std::string;
    std::unordered_map<std::string, int> vocab;
    std::unordered_map<int, std::string> id_to_token;
    std::unordered_map<uint64_t, int> merges;
    std::vector<std::string> added_tokens;
    ByteEncoder encoder;

    int bos_token_id;
    int pad_token_id;
    std::vector<int> eos_token_id;
    float temperature;
    int top_k;
    float top_p;

    std::vector<std::string> byte_split(const std::string& text);
    int find_best_merge(const std::vector<std::string>& tokens, int& best_idx);
    std::vector<std::string> bpe(const std::string& text);
    std::vector<int> encode(const std::string& text);
    void load_vocab(const char* path);
    void load_merges(const char* path);
    std::string decode(int id);
    void load_gen_config(const char* path);
    void load_tokenizer_config(const char* path);
    std::string apply_chat_template(
        const std::vector<ChatMessage>& messages,
        bool add_generation_prompt = true,
        const std::vector<ToolDefinition>& tools = {},
        bool enable_thinking = false
    ) const;
};


struct Sampler {
    float m_temperature;
    int m_top_k;
    float m_top_p;
    
    std::vector<int> m_indices;     

    Sampler(int vocab_size,float t,float top_p, int top_k);
    
    /*if do_sample is false, it will do greedy sampling (i.e., argmax)
      if do_sample is true, it will do multinomial sampling with temperature, top_k and top_p
    */
    int sample(const Tensor* logits, bool do_sample=true);
    int sample_multinomial(const std::vector<int>& idx,
                           const std::vector<float>& p);
};
