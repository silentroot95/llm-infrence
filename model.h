#pragma once

#include "tensor.h"


struct ModelConfig {
    int max_seq_len;    //max sequence length
    int embed_size;     //embedding size
    int q_heads;        // Query heads
    int kv_heads;       // Key value heads
    int head_size;
    int num_layers;     //attention layers
    int ffn_size;       //feed forward hidden size
    int vocab_size;
    float rms_norm_eps;
    int bos_token_id;
    int eos_token_id;

    float rope_theta;

    void load_config(const char* path);
};

struct TransformerLayer {
    Tensor attn_norm;       //attention layer norm
    Tensor q_weight;        //q project
    Tensor k_weight;        //k project
    Tensor v_weight;        //v project
    Tensor o_weight;        //o project
    Tensor q_norm;          //q norm
    Tensor k_norm;          //k norm
    Tensor ffn_norm;        //ffn layer norm
    Tensor up_weight;       //ffn up weight
    Tensor gate_weight;     //ffn gate weight
    Tensor down_weight;     //ffn down weight
};

struct ModelWeights {
    Tensor embeds;
    Tensor output_norm;
    QuantizedWeightINT8 lm_heads;
    TransformerLayer* layers;
    void* data;

    int num_layers;

    void init(const ModelConfig* config);
    inline void destroy();
    void load_tensor(const char* path);
};

inline void ModelWeights::destroy() {
    free(layers);
    free(data);
}

struct RunTimeMemory {
    void* k_data;            // [layers,seq_len_max,kv_heads,head_size]
    void* v_data;            // [layers,seq_len_max,kv_heads,head_size]

    Tensor k_cache;          //  [seq_len_max,kv_heads,head_size]
    Tensor v_cache;          //  [seq_len_max,kv_heads,head_size]
    
    Tensor in;
    Tensor out;

    Tensor q;                
    Tensor k;                //  [seq_len_inprocess,kv_heads,head_size]
    Tensor v;                //  [seq_len_inprocess,kv_heads,head_size]

    Tensor up;
    Tensor gate;

    Tensor final_logits;     //[1, vocab_size]

    int seq_len_processed;

    int prefill_chunck;      // for prefill, process tokens in chunck to save memory
    int max_seq_len;

    uint8_t* ptr;            // temp buffer in runtime
    size_t offset;           // current offset in temp buffer
    size_t capacity;         // total size of temp buffer

    void init(const ModelConfig* config, int len);
    void destroy();
    void* allocate(size_t size,size_t align = 64);
    void* allocate_kv(int li, int seq_len, const ModelConfig* config,bool is_k);
    void reset();
    void update_len(int len);

};

inline void* RunTimeMemory::allocate(size_t size, size_t align) {
    size_t p = (offset + align - 1) & ~(align - 1);
    if (p + size > capacity) {
        fprintf(stderr, "OOM: offset=%zu size=%zu capacity=%zu\n", p, size, capacity);
        exit(1);
    }
    void* res = ptr + p;
    offset = p + size;
    return res;
}

inline void RunTimeMemory::reset() {
    offset = 0;
}

inline void RunTimeMemory::update_len(int len) {
    in.shape[0] = out.shape[0] = len;
    q.shape[0] = k.shape[0] = v.shape[0] = len;
    up.shape[0] = gate.shape[0] = len;

    k_cache.shape[0] = v_cache.shape[0] = seq_len_processed + len;
}

struct Header {
    int dim1;
    int dim2;
};

void save_logits(const char* filename, float* logits, int m, int n);


