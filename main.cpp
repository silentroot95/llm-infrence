#include "qwen3.h"
#include "tokenizer.h"

#include <chrono>

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
    //printf("%d",tk.vocab.size());
    tk.load_tokenizer_config("./qwen3-0.6b/tokenizer_config.json");
    tk.load_merges("./qwen3-0.6b/merges.txt");
    tk.load_gen_config("./qwen3-0.6b/generation_config.json");

    Qwen3 qwen3 = Qwen3(&model_config, &model_weights, &memory);
    

    std::string user_prompt = "阅读下面的材料，根据要求写作。\n"
                             "他想要给孩子们唱上一段，可是心里直翻腾，开不了口。\n"
                             "——老舍《鼓书艺人》（见全国一卷阅读II）\n"
                             "假如我是一只鸟，\n"
                             "我也应该用嘶哑的喉咙歌唱\n"
                             "——艾青《我爱这土地》\n"
                             "我要以带血的手和你们一一拥抱，\n"
                             "因为一个民族已经起来\n"
                             "——穆旦《赞美》\n"
                             "以上材料引发了你怎样的联想和思考？请写一篇文章。\n"
                             "要求：选准角度，确定立意，明确文体，自拟标题；不要套作，不得抄袭；不得泄露个人信息；不少于800字。";

    std::vector<ChatMessage> messages = {
       // {"system", "You are a helpful assistant.", "", {}},
        {"user", user_prompt, "", {}}
    };

    std::string prompt = tk.apply_chat_template(messages, true, {}, true);
    //printf("Prompt:\n%s\n", prompt.c_str());
    std::vector<int> token_ids = tk.encode(prompt);

    Sampler sampler(model_config.vocab_size, 0.6,0.95,20);


    Tensor chunk;
    chunk.data_type = TENSOR_DATA_TYPE_INT32;
    chunk.dim = 1;
    
    Tensor* logits = nullptr;
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
    return 0;
}
