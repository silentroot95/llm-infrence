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
    Tensor lm_heads;
    TransformerLayer* layers;
    void* data;

    int num_layers;

    void init(const ModelConfig* config);
    inline void destory();
    void pack_hot_weights();
    void load_tensor(const char* path);
};


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

    int seq_len_processed;

    int max_prefill_len;
    int max_seq_len;

    uint8_t* ptr;            // temp buffer in runtime
    int offset;              // ptr - ptr_begin = bytes in use

    void init(const ModelConfig* config, int len);
    void destroy();
    void* allocate(size_t size);
    void* allocate_kv(int li, int seq_len, const ModelConfig* config,bool is_k);
    void reinit(const ModelConfig* config, int len);
    void reset();
    void update_len(int len);
    //RunTimeMemory(const RunTimeMemory&)=delete;
    //RunTimeMemory& operator=(const RunTimeMemory&)=delete;
};



inline void* RunTimeMemory::allocate(size_t size) {
    void* res = (void*)(ptr + offset);
    offset += size;
    return res;
}

inline void RunTimeMemory::reset() {
    offset = 0;
}


inline void RunTimeMemory::update_len(int len) {
    q.shape[0] = k.shape[0] = v.shape[0] = len;
    up.shape[0] = gate.shape[0] = len;
}
