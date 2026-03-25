#pragma once

#include "model.h"
#include "operator.h"

#include <chrono>

enum ProfileOp {
    PROFILE_EMBED = 0,
    PROFILE_RMS_NORM,
    PROFILE_MATMUL_QKV,
    PROFILE_MATMUL_O,
    PROFILE_MATMUL_FFN_UP_GATE,
    PROFILE_MATMUL_FFN_DOWN,
    PROFILE_MATMUL_LM_HEAD,
    PROFILE_ROPE,
    PROFILE_ATTENTION,
    PROFILE_ADD,
    PROFILE_MLP,
    PROFILE_STORE_BF16,
    PROFILE_OP_COUNT
};

struct OpProfileStats {
    double total_ms[PROFILE_OP_COUNT];
    int calls[PROFILE_OP_COUNT];

    inline void reset() {
        for (int i = 0; i < PROFILE_OP_COUNT; ++i) {
            total_ms[i] = 0.0;
            calls[i] = 0;
        }
    }
};

static inline const char* profile_op_name(ProfileOp op) {
    switch (op) {
        case PROFILE_EMBED: return "embed";
        case PROFILE_RMS_NORM: return "rms_norm";
        case PROFILE_MATMUL_QKV: return "matmul_qkv";
        case PROFILE_MATMUL_O: return "matmul_o";
        case PROFILE_MATMUL_FFN_UP_GATE: return "matmul_ffn_up_gate";
        case PROFILE_MATMUL_FFN_DOWN: return "matmul_ffn_down";
        case PROFILE_MATMUL_LM_HEAD: return "matmul_lm_head";
        case PROFILE_ROPE: return "rope";
        case PROFILE_ATTENTION: return "attention";
        case PROFILE_ADD: return "add";
        case PROFILE_MLP: return "mlp";
        case PROFILE_STORE_BF16: return "store_bf16";
        default: return "unknown";
    }
}

template <typename Fn>
static inline void profile_run(OpProfileStats* stats, ProfileOp op, Fn&& fn) {
    auto start = std::chrono::high_resolution_clock::now();
    fn();
    auto end = std::chrono::high_resolution_clock::now();
    stats->total_ms[op] += std::chrono::duration<double, std::milli>(end - start).count();
    stats->calls[op] += 1;
}


struct Qwen3 {
    const ModelWeights* model_weights;
    const ModelConfig* model_config;
    RunTimeMemory* mem;
    OpProfileStats profile_stats;

    Qwen3(const ModelConfig* c, const ModelWeights* w, RunTimeMemory* m);

    Tensor* forward(const Tensor* tokenid);
    inline void reset_profile();
    inline void print_profile(const char* stage_name) const;
};

Qwen3::Qwen3(const ModelConfig* c,const ModelWeights* w, RunTimeMemory* m) :
    model_config    (c), 
    model_weights   (w),
    mem             (m)
    {
        profile_stats.reset();
    }

inline void Qwen3::reset_profile() {
    profile_stats.reset();
}

inline void Qwen3::print_profile(const char* stage_name) const {
    double total_ms = 0.0;
    for (int i = 0; i < PROFILE_OP_COUNT; ++i) {
        total_ms += profile_stats.total_ms[i];
    }

    printf("[Profile] %s total op time: %.2f ms\n", stage_name, total_ms);
    for (int i = 0; i < PROFILE_OP_COUNT; ++i) {
        if (profile_stats.calls[i] == 0) {
            continue;
        }
        double ms = profile_stats.total_ms[i];
        double pct = total_ms > 0.0 ? (100.0 * ms / total_ms) : 0.0;
        printf("[Profile] %-10s %8.2f ms  %6.2f%%  calls=%d\n",
               profile_op_name((ProfileOp)i), ms, pct, profile_stats.calls[i]);
    }
}


Tensor* Qwen3::forward(const Tensor* tokenid) {
    int seq_len_inprocess = tokenid->shape[0];
    mem->update_len(seq_len_inprocess);

    profile_run(&profile_stats, PROFILE_EMBED, [&](){
        embed(tokenid,&model_weights->embeds,&mem->in);
    });

    Tensor kv_cache_dst;
    kv_cache_dst.shape[0] = seq_len_inprocess;
    kv_cache_dst.shape[1] = model_config->kv_heads * model_config->head_size;
    kv_cache_dst.data_type = mem->k_cache.data_type;
    kv_cache_dst.dim = 2;

    for(int i=0; i<model_config->num_layers;++i) {
        profile_run(&profile_stats, PROFILE_RMS_NORM, [&](){
            rms_norm(&mem->in,
                     &model_weights->layers[i].attn_norm,
                     &mem->out,
                     model_config->rms_norm_eps);
        });

        //Q,K,V project
        mem->reset();
        //q [seqlen, q_head*head_size], q allocate from temp buffer
        mem->q.data = mem->allocate(model_config->q_heads * 
                                    model_config->head_size *
                                    seq_len_inprocess * sizeof(float));
        // k v use f32 temp buffers first, then store to bf16 cache after rope.
        mem->k.data = mem->allocate(model_config->kv_heads *
                                    model_config->head_size *
                                    seq_len_inprocess * sizeof(float));
        mem->v.data = mem->allocate(model_config->kv_heads *
                                    model_config->head_size *
                                    seq_len_inprocess * sizeof(float));

        profile_run(&profile_stats, PROFILE_MATMUL_QKV, [&](){
            matmul(&mem->out,
                    &model_weights->layers[i].k_weight,
                    &mem->k);
        });
        profile_run(&profile_stats, PROFILE_MATMUL_QKV, [&](){
            matmul(&mem->out,
                    &model_weights->layers[i].v_weight,
                    &mem->v);
        });
        profile_run(&profile_stats, PROFILE_MATMUL_QKV, [&](){
            matmul(&mem->out,
                    &model_weights->layers[i].q_weight,
                    &mem->q);
        });

        //Q,K,V reshape to n_head*head_size
        mem->k.reshape_3d(model_config->head_size);
        mem->q.reshape_3d(model_config->head_size);
        mem->v.reshape_3d(model_config->head_size);
        
        //Q,K norm inplace
        profile_run(&profile_stats, PROFILE_RMS_NORM, [&](){
            rms_norm(&mem->k, 
                    &model_weights->layers[i].k_norm,
                    &mem->k,
                    model_config->rms_norm_eps);
        });

        profile_run(&profile_stats, PROFILE_RMS_NORM, [&](){
            rms_norm(&mem->q, 
                    &model_weights->layers[i].q_norm,
                    &mem->q,
                    model_config->rms_norm_eps);
        });

        //Q,K rope inplace
        profile_run(&profile_stats, PROFILE_ROPE, [&](){
            rope(&mem->k,
                model_config->rope_theta,
                mem->seq_len_processed);
        });

        profile_run(&profile_stats, PROFILE_ROPE, [&](){
            rope(&mem->q,
                model_config->rope_theta,
                mem->seq_len_processed);
        });

        kv_cache_dst.data = mem->allocate_kv(i, seq_len_inprocess, model_config, true);
        profile_run(&profile_stats, PROFILE_STORE_BF16, [&](){
            store_f32_tensor_to_bf16(&mem->k, &kv_cache_dst);
        });

        kv_cache_dst.data = mem->allocate_kv(i, seq_len_inprocess, model_config, false);
        profile_run(&profile_stats, PROFILE_STORE_BF16, [&](){
            store_f32_tensor_to_bf16(&mem->v, &kv_cache_dst);
        });
        
        //attn_score shape [q_head, seq_len], for attention score of one token 
        float* att_score = (float*)mem->allocate(model_config->q_heads*
                                                (seq_len_inprocess + mem->seq_len_processed) * sizeof(float));
        //QKV multi head attention
        //mem->out now is free so we can reuse it
        profile_run(&profile_stats, PROFILE_ATTENTION, [&](){
            attention(  &mem->q,
                        &mem->k_cache,
                        &mem->v_cache,
                        mem->seq_len_processed,
                        att_score);
        });
        profile_run(&profile_stats, PROFILE_MATMUL_O, [&](){
            matmul(  &mem->q,
                     &model_weights->layers[i].o_weight,
                     &mem->out);
        });
        
        //residual connect
        profile_run(&profile_stats, PROFILE_ADD, [&](){
            add_inplace(&mem->in,&mem->out);
        });
        
        //ffn norm
        profile_run(&profile_stats, PROFILE_RMS_NORM, [&](){
            rms_norm(&mem->in,
                     &model_weights->layers[i].ffn_norm,
                     &mem->out,
                     model_config->rms_norm_eps);
        });
        
        //mem->up mem->gate  data is in temp buffer
        mem->reset();
        mem->up.data = mem->allocate(model_config->ffn_size * 
                                    seq_len_inprocess * sizeof(float));
        mem->gate.data = mem->allocate(model_config->ffn_size * 
                                    seq_len_inprocess * sizeof(float));

        profile_run(&profile_stats, PROFILE_MATMUL_FFN_UP_GATE, [&](){
            matmul_pair(&mem->out,
                        &model_weights->layers[i].up_weight,
                        &mem->up,
                        &model_weights->layers[i].gate_weight,
                        &mem->gate);
        });
        
        profile_run(&profile_stats, PROFILE_MLP, [&](){
            mlp(&mem->up,
                &mem->gate);
        });
        profile_run(&profile_stats, PROFILE_MATMUL_FFN_DOWN, [&](){
            matmul( &mem->up,
                    &model_weights->layers[i].down_weight,
                    &mem->out);
        });
        
        //residual connect
        profile_run(&profile_stats, PROFILE_ADD, [&](){
            add_inplace(&mem->in,&mem->out);
        });
    }

    //result norm
    profile_run(&profile_stats, PROFILE_RMS_NORM, [&](){
        rms_norm(&mem->in,
                 &model_weights->output_norm,
                 &mem->in,
                 model_config->rms_norm_eps);
    });

    //final logits, we only get the last token output
    Tensor last = mem->in.peek_at(mem->in.shape[0]-1);

    profile_run(&profile_stats, PROFILE_MATMUL_LM_HEAD, [&](){
        matmul( &last,
                &model_weights->lm_heads,
                &mem->out);
    });

    mem->seq_len_processed += seq_len_inprocess;

    return &(mem->out);
}


/*
void Qwen3::build_graph(RunTimeMemory& mem,const ModelWeights& model_weights) {
    //Tensor* in = mem->allocate_out();
    build_embed(nullptr,&model_weights->embeds,&mem->in);

        
    for(int i=0;i<model_weights->num_layers;++i)
    {
        //input norm
        //Tensor* norm = mem->allocate_out();
        build_norm(&mem->in,&model_weights->layers[i].attn_norm,&mem->out);

        //Q,K,V project
        build_matmul(&mem->out,&model_weights->layers[i].k_weight,&mem->k);
        build_matmul(&mem->out,&model_weights->layers[i].v_weight,&mem->v);
        build_matmul(&mem->out,&model_weights->layers[i].q_weight,&mem->q);

        //Q,K,V reshape to n_head*head_size
        build_reshape_3d(&mem->k, model_weights->layers[i].k_norm.shape[0]);
        build_reshape_3d(&mem->q, model_weights->layers[i].q_norm.shape[0]);
        build_reshape_3d(&mem->v, model_weights->layers[i].k_norm.shape[0]);
        
        //Q,K norm inplace
        build_norm(&mem->k, &model_weights->layers[i].k_norm,&mem->k);
        build_norm(&mem->q, &model_weights->layers[i].q_norm,&mem->q);

        //Q,K rope inplace
        build_rope(&mem->k,&mem->pos);
        build_rope(&mem->q,&mem->pos);

        //QKV multi head attention
        //Tensor* out = mem->allocate_out();
        build_attention(&mem->q,&mem->k,&mem->v,&mem->att_score,&mem->out);
        
        //residual connect
        build_add(&mem->in,&mem->out,&mem->in);

        //ffn norm
        build_norm(&mem->in,&model_weights->layers[i].ffn_norm,&mem->out);
        
        build_matmul(&mem->out,&model_weights->layers[i].up_weight,&mem->up);

        build_matmul(&mem->out,&model_weights->layers[i].gate_weight,&mem->gate);

        //FFN
        build_mlp(&mem->out,&mem->up,&mem->gate,&mem->ffn_out);
        
        //residual connect
        build_add(&mem->in,&mem->ffn_out,&mem->in);
    }
    
    //result norm
    build_norm(&mem->in,&model_weights->output_norm,&mem->in);

    //final logits
    build_matmul(&mem->in,&model_weights->lm_heads,&mem->out);
}
*/
