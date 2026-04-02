#include "qwen3.h"
#include "tokenizer.h"

#include <iostream>
#include <chrono>
#include <cctype>

bool is_empty_or_whitespace(const std::string& s) {
    for (unsigned char c : s) {
        if (!std::isspace(c)) {
            return false;
        }
    }
    return true;
}
inline bool is_eos(int token, const std::vector<int>& eos_ids) {
    for (int id : eos_ids) {
        if (token == id) return true;
    }
    return false;
}

inline void update_next_token(int next_token, Tensor* tokens) {
    tokens->shape[0] = 1;
    int* data = (int*)tokens->data;
    data[0] = next_token;
}


int main() {
    ModelConfig model_config;
    model_config.load_config("./qwen3-0.6b/config.json");

    ModelWeights model_weights;
    model_weights.init(&model_config);
    model_weights.load_tensor("./qwen3-0.6b/model.safetensors");

    RunTimeMemory memory;
    int prefill_chunk = 1024;
    memory.init(&model_config,prefill_chunk);

    Tokenizer tk;
    tk.load_vocab("./qwen3-0.6b/vocab.json");
    tk.load_tokenizer_config("./qwen3-0.6b/tokenizer_config.json");
    tk.load_merges("./qwen3-0.6b/merges.txt");
    tk.load_gen_config("./qwen3-0.6b/generation_config.json");

    Qwen3 qwen3 = Qwen3(&model_config, &model_weights, &memory);
    Sampler sampler(model_config.vocab_size, 0.6,0.95,20);

    Tensor chunk;
    chunk.data_type = TENSOR_DATA_TYPE_INT32;
    chunk.dim = 1;

    while (true) {
        std::string user_prompt;
        Tensor* logits = nullptr;
        memory.seq_len_processed = 0;

        printf("\nUser> ");
        std::getline(std::cin, user_prompt);

        if(is_empty_or_whitespace(user_prompt)) continue;

        if (user_prompt == "/exit") break;

        std::vector<ChatMessage> messages = {
            {"user", user_prompt, "", {}}
        };

        std::string prompt = tk.apply_chat_template(messages, true, {}, true);
        std::vector<int> token_ids = tk.encode(prompt);

        //prefill
        qwen3.reset_profile();
        int processed_tokens = 0;
        auto step_start = std::chrono::high_resolution_clock::now();
        while(1) {
            int chunk_size = std::min(prefill_chunk, (int)token_ids.size() - processed_tokens);
            chunk.shape[0] = chunk_size;
            chunk.data = (void*)(token_ids.data() + processed_tokens);
            logits = qwen3.forward(&chunk);
            processed_tokens += chunk_size;
            
            if(processed_tokens >= token_ids.size()) {
                auto step_end = std::chrono::high_resolution_clock::now();
                double prefill_ms = std::chrono::duration<double, std::milli>(step_end - step_start).count();
                double prefill_tps = prefill_ms > 0.0 ? (1000.0 * processed_tokens / prefill_ms) : 0.0;

                printf("\n\n[Perf] prefill: %d tokens, %.2f ms, %.2f tokens/s\n",
                    processed_tokens, prefill_ms, prefill_tps);
                qwen3.print_profile("prefill");
                break;
            }
        }

        //decode
        int decode_tokens = 0;
        double decode_ms = 0.0;
        qwen3.reset_profile();
        while (1) {
            int next_token = sampler.sample(logits, true);
            if (is_eos(next_token, tk.eos_token_id) || 
                memory.seq_len_processed >= model_config.max_seq_len) {
                break;
            }

            std::string next_s = tk.decode(next_token);
            printf("%s",next_s.c_str());
            fflush(stdout);
            update_next_token(next_token, &chunk);

            auto step_start = std::chrono::high_resolution_clock::now();
            logits = qwen3.forward(&chunk);
            auto step_end = std::chrono::high_resolution_clock::now();
            decode_ms += std::chrono::duration<double, std::milli>(step_end - step_start).count();
            decode_tokens += 1;
        }

        double decode_tps = decode_ms > 0.0 ? (1000.0 * decode_tokens / decode_ms) : 0.0;
        printf("\n[Perf] decode : %d tokens, %.2f ms, %.2f tokens/s\n",
            decode_tokens, decode_ms, decode_tps);
        qwen3.print_profile("decode");

    }

    model_weights.destroy();
    memory.destroy();
    return 0;
}
