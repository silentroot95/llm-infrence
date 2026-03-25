#pragma once

#include <vector>
#include <unordered_map>

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


struct Tokenizer {
    //using std::string;
    std::unordered_map<std::string, int> vocab;
    std::unordered_map<int, std::string> id_to_token;
    std::unordered_map<uint64_t, int> merges;
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
    void encode(const std::string& text, Tensor& ids);
    void load_vocab(const char* path);
    void load_merges(const char* path);
    std::string decode(int id);
    void load_gen_config(const char* path);
};


struct Sampler {
    float temperature;
    int top_k;
    float top_p;
    
    std::vector<int> indices;     
    std::vector<float> probs;

    Sampler(int vocab_size,float t,int top_k,float top_p);
    int sample(Tensor* logits);
    int sample_multinomial(const std::vector<int>& idx,
                           const std::vector<float>& p);
};