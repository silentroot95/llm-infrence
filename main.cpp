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
    

    std::string user_prompt = ("你是一个高性能推理引擎的内部调试模块，请严格按照以下规则进行“思考模拟”但不要输出推理过程，只输出最终结构化结果。本任务用于测试系统在复杂上下文下的prefill性能，因此输入文本较长且包含多种语义结构。\n"
                          "输入文本包含多轮对话，每轮对话由一个角色（system、user、assistant）和对应的内容组成。内容中可能包含纯文本、JSON结构化数据、以及特殊标记等。\n"
                          "请根据输入的对话内容，模拟模型在prefill阶段的思考过程，输出一个JSON对象，包含以下字段：\n"
                          "1. tokens: 模拟prefill阶段模型接收到的token序列（以token id形式表示）。\n"
                          "2. attention_pattern: 模拟prefill阶段模型计算的注意力模式，描述哪些token之间会有较强的注意力连接。\n"
                          "3. kv_cache_update: 模拟prefill阶段模型更新的KV缓存状态，描述哪些token被存储到KV缓存中，以及它们对应的层数和位置。\n"
                          "4. generation_prompt: 如果add_generation_prompt为true，模拟prefill阶段模型生成的下一步生成提示（generation prompt），描述它的内容和位置。\n"
                          "请确保输出的JSON对象格式正确，并且内容详尽地反映了模型在prefill阶段可能的内部状态和思考过程。");


    std::vector<ChatMessage> messages = {
       // {"system", "You are a helpful assistant.", "", {}},
        {"user", "解释下什么是Transformer模型？", "", {}}
    };

    std::string prompt = tk.apply_chat_template(messages, true, {}, true);
    printf("Prompt:\n%s\n", prompt.c_str());
    std::vector<int> token_ids = tk.encode(prompt);

    for (int id : token_ids) {
        printf("%d ", id);
    }
    printf("\n");

    Sampler sampler(model_config.vocab_size, 0.7,0.8,20);


    Tensor tokens;
    tokens.data_type = TENSOR_DATA_TYPE_INT32;
    tokens.dim = 1;
    tokens.shape[0] = token_ids.size();
    tokens.data = (void*)token_ids.data();
    int prefill_tokens = tokens.shape[0];

    qwen3.reset_profile();
    auto step_start = std::chrono::high_resolution_clock::now();
    Tensor* logits = qwen3.forward(&tokens);
    auto step_end = std::chrono::high_resolution_clock::now();
    double prefill_ms = std::chrono::duration<double, std::milli>(step_end - step_start).count();
    double prefill_tps = prefill_ms > 0.0 ? (1000.0 * prefill_tokens / prefill_ms) : 0.0;

    printf("\n\n[Perf] prefill: %d tokens, %.2f ms, %.2f tokens/s\n",
           prefill_tokens, prefill_ms, prefill_tps);
    qwen3.print_profile("prefill");
    
    int decode_tokens = 0;
    double decode_ms = 0.0;  
    qwen3.reset_profile();

    while (1) {
        int next_token  = sampler.sample(logits, true);

        if (is_eos(next_token, tk.eos_token_id) || 
            memory.seq_len_processed >= model_config.max_seq_len ||
            decode_tokens >= 256) {
            break;
        }

        std::string next_s = tk.decode(next_token);
        printf("%s",next_s.c_str());
        fflush(stdout);
        update_next_token(next_token, &tokens);

        step_start = std::chrono::high_resolution_clock::now();
        logits = qwen3.forward(&tokens);
        step_end = std::chrono::high_resolution_clock::now();
        decode_ms += std::chrono::duration<double, std::milli>(step_end - step_start).count();
        decode_tokens += 1;
    }

    double decode_tps = decode_ms > 0.0 ? (1000.0 * decode_tokens / decode_ms) : 0.0;
    printf("\n[Perf] decode : %d tokens, %.2f ms, %.2f tokens/s\n",
           decode_tokens, decode_ms, decode_tps);
    qwen3.print_profile("decode");
    return 0;
}
