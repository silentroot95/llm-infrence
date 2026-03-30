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

    Tensor* forward(const Tensor* token_ids);
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

Tensor* Qwen3::forward(const Tensor* token_ids) {
    int seq_len_inprocess = token_ids->shape[0];
    //mem->update_len(seq_len_inprocess);
    char logit_filename[256];

    mem->in.shape[0] = seq_len_inprocess;
    mem->in.shape[1] = model_config->embed_size;
    mem->in.dim = 2;
    mem->in.data_type = TENSOR_DATA_TYPE_F32;

    profile_run(&profile_stats, PROFILE_EMBED, [&](){
        embed(token_ids,&model_weights->embeds,&mem->in);
    });

    #define DEBUG 1

    mem->v_cache.shape[0] = mem->k_cache.shape[0] = mem->seq_len_processed + seq_len_inprocess;
    mem->v_cache.shape[1] = mem->k_cache.shape[1] = model_config->kv_heads;
    mem->v_cache.shape[2] = mem->k_cache.shape[2] = model_config->head_size;
    mem->v_cache.dim = mem->k_cache.dim = 3;
    mem->v_cache.data_type = mem->k_cache.data_type = TENSOR_DATA_TYPE_F32;

    for(int i=0; i<model_config->num_layers;++i) {
        
        mem->out.copy_meta(&mem->in);

        profile_run(&profile_stats, PROFILE_RMS_NORM, [&](){
            rms_norm(&mem->in,
                     &model_weights->layers[i].attn_norm,
                     &mem->out,
                     model_config->rms_norm_eps);
        });

        #if defined(DEBUG)
        snprintf(logit_filename, sizeof(logit_filename), "./cppbin/input_%d.bin", i);
        save_logits(logit_filename, (float*)mem->in.data, mem->in.shape[0], mem->in.shape[1]);
        snprintf(logit_filename, sizeof(logit_filename), "./cppbin/input_norm_%d.bin", i);
        save_logits(logit_filename, (float*)mem->out.data, mem->out.shape[0], mem->out.shape[1]);
        #endif

        //Q,K,V project
        mem->reset();
        //q [seqlen, q_head*head_size], q allocate from temp buffer
        mem->q.data = mem->allocate(model_config->q_heads * 
                                    model_config->head_size *
                                    seq_len_inprocess * sizeof(float));
        // k v use f32 temp buffers first, then store to bf16 cache after rope.
        /*
        mem->k.data = mem->allocate(model_config->kv_heads *
                                    model_config->head_size *
                                    seq_len_inprocess * sizeof(float));
        mem->v.data = mem->allocate(model_config->kv_heads *
                                    model_config->head_size *
                                    seq_len_inprocess * sizeof(float));
        */

        mem->k.data = mem->allocate_kv(i, seq_len_inprocess, model_config, true);
        mem->v.data = mem->allocate_kv(i, seq_len_inprocess, model_config, false);

        mem->q.copy_meta(&mem->out);
        mem->k.copy_meta(&mem->out);
        mem->v.copy_meta(&mem->out);

        mem->q.shape[1] = model_config->q_heads * model_config->head_size;
        mem->v.shape[1] = mem->k.shape[1] = model_config->kv_heads * model_config->head_size;


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

        #if defined(DEBUG)
        snprintf(logit_filename, sizeof(logit_filename), "./cppbin/k_%d.bin", i);
        save_logits(logit_filename, (float*)mem->k.data, mem->k.shape[0], mem->k.shape[1]);
        snprintf(logit_filename, sizeof(logit_filename), "./cppbin/v_%d.bin", i);
        save_logits(logit_filename, (float*)mem->v.data, mem->v.shape[0], mem->v.shape[1]);
        snprintf(logit_filename, sizeof(logit_filename), "./cppbin/q_%d.bin", i);
        save_logits(logit_filename, (float*)mem->q.data, mem->q.shape[0], mem->q.shape[1]);
        #endif

        //Q,K,V reshape to n_head*head_size
        mem->k.reshape_3d(model_config->head_size);
        mem->q.reshape_3d(model_config->head_size);
        mem->v.reshape_3d(model_config->head_size);
        
        //Q,K norm inplace
        profile_run(&profile_stats, PROFILE_RMS_NORM, [&](){
            rms_norm(&mem->k, 
                    &model_weights->layers[i].k_norm,
                    nullptr,
                    model_config->rms_norm_eps);
        });

        profile_run(&profile_stats, PROFILE_RMS_NORM, [&](){
            rms_norm(&mem->q, 
                    &model_weights->layers[i].q_norm,
                    nullptr,
                    model_config->rms_norm_eps);
        });

        #if defined(DEBUG)
        snprintf(logit_filename, sizeof(logit_filename), "./cppbin/k_norm_%d.bin", i);
        save_logits(logit_filename, (float*)mem->k.data, mem->k.shape[0], mem->k.shape[1]*mem->k.shape[2]);
        snprintf(logit_filename, sizeof(logit_filename), "./cppbin/q_norm_%d.bin", i);
        save_logits(logit_filename, (float*)mem->q.data, mem->q.shape[0], mem->q.shape[1]*mem->q.shape[2]);
        #endif

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

        #if defined(DEBUG)
        snprintf(logit_filename, sizeof(logit_filename), "./cppbin/k_rope_%d.bin", i);
        save_logits(logit_filename, (float*)mem->k.data, mem->k.shape[0], mem->k.shape[1]*mem->k.shape[2]);
        snprintf(logit_filename, sizeof(logit_filename), "./cppbin/q_rope_%d.bin", i);
        save_logits(logit_filename, (float*)mem->q.data, mem->q.shape[0], mem->q.shape[1]*mem->q.shape[2]);
        #endif

        /**
        kv_cache_dst.data = mem->allocate_kv(i, seq_len_inprocess, model_config, true);
        profile_run(&profile_stats, PROFILE_STORE_BF16, [&](){
            store_f32_tensor_to_bf16(&mem->k, &kv_cache_dst);
        });

        kv_cache_dst.data = mem->allocate_kv(i, seq_len_inprocess, model_config, false);
        profile_run(&profile_stats, PROFILE_STORE_BF16, [&](){
            store_f32_tensor_to_bf16(&mem->v, &kv_cache_dst);
        });
         */
        
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
        
        mem->q.reshape_2d();


        mem->out.copy_meta(&mem->q);
        mem->out.shape[1] = model_config->embed_size;

        profile_run(&profile_stats, PROFILE_MATMUL_O, [&](){
            matmul(  &mem->q,
                     &model_weights->layers[i].o_weight,
                     &mem->out);
        });

        #if defined(DEBUG)
        snprintf(logit_filename, sizeof(logit_filename), "./cppbin/attnout_%d.bin", i);
        save_logits(logit_filename, (float*)mem->q.data, mem->q.shape[0], mem->q.shape[1]);

        snprintf(logit_filename, sizeof(logit_filename), "./cppbin/o_%d.bin", i);
        save_logits(logit_filename, (float*)mem->out.data, mem->out.shape[0], mem->out.shape[1]);
        #endif

        //residual connect
        profile_run(&profile_stats, PROFILE_ADD, [&](){
            add_inplace(&mem->in,&mem->out);
        });
        
        mem->out.copy_meta(&mem->in);

        //ffn norm
        profile_run(&profile_stats, PROFILE_RMS_NORM, [&](){
            rms_norm(&mem->in,
                     &model_weights->layers[i].ffn_norm,
                     &mem->out,
                     model_config->rms_norm_eps);
        });

        #if defined(DEBUG)
        snprintf(logit_filename, sizeof(logit_filename), "./cppbin/ffn_norm_in_%d.bin", i);
        save_logits(logit_filename, (float*)mem->in.data, mem->in.shape[0], mem->in.shape[1]);
        snprintf(logit_filename, sizeof(logit_filename), "./cppbin/ffn_norm_out_%d.bin", i);
        save_logits(logit_filename, (float*)mem->out.data, mem->out.shape[0], mem->out.shape[1]);
        #endif
        
        
        /**
        profile_run(&profile_stats, PROFILE_MATMUL_FFN_UP_GATE, [&](){
            matmul_pair(&mem->out,
                        &model_weights->layers[i].up_weight,
                        &mem->up,
                        &model_weights->layers[i].gate_weight,
                        &mem->gate);
        });
         */

        
        //mem->up mem->gate  data is in temp buffer
        mem->reset();

        size_t up_gate_size = model_config->ffn_size * seq_len_inprocess * sizeof(float);
        mem->up.data = mem->allocate(up_gate_size);
        mem->gate.data = mem->allocate(up_gate_size);

        mem->gate.copy_meta(&mem->out);
        mem->up.copy_meta(&mem->out);
        mem->up.shape[1] = mem->gate.shape[1] = model_config->ffn_size;

        /*
        profile_run(&profile_stats, PROFILE_MATMUL_FFN_UP_GATE, [&](){
            matmul( &mem->out,
                    &model_weights->layers[i].gate_weight,
                    &mem->gate);
        });
        */

        matmul( &mem->out,
                            &model_weights->layers[i].gate_weight,
                            &mem->gate);

        snprintf(logit_filename, sizeof(logit_filename), "./cppbin/gate_%d.bin", i);
        save_logits(logit_filename, (float*)mem->gate.data, mem->gate.shape[0], mem->gate.shape[1]);

        matmul( &mem->out,
                    &model_weights->layers[i].up_weight,
                    &mem->up);



        snprintf(logit_filename, sizeof(logit_filename), "./cppbin/up_%d.bin", i);
        save_logits(logit_filename, (float*)mem->up.data, mem->up.shape[0], mem->up.shape[1]);

        /*
        profile_run(&profile_stats, PROFILE_MATMUL_FFN_UP_GATE, [&](){
            matmul( &mem->out,
                    &model_weights->layers[i].up_weight,
                    &mem->up);
        });
        */  
        /**
        profile_run(&profile_stats, PROFILE_MLP, [&](){
            mlp(&mem->up,
                &mem->gate);
        });
         */
        mlp(&mem->up,
                &mem->gate);

        #if defined(DEBUG)
        snprintf(logit_filename, sizeof(logit_filename), "./cppbin/ug_%d.bin", i);
        save_logits(logit_filename, (float*)mem->up.data, mem->up.shape[0], mem->up.shape[1]);
        #endif

        mem->out.copy_meta(&mem->up);
        mem->out.shape[1] = model_config->embed_size;

        profile_run(&profile_stats, PROFILE_MATMUL_FFN_DOWN, [&](){
            matmul( &mem->up,
                    &model_weights->layers[i].down_weight,
                    &mem->out);
        });

        #if defined(DEBUG)
        snprintf(logit_filename, sizeof(logit_filename), "./cppbin/ffn_out_%d.bin", i);
        save_logits(logit_filename, (float*)mem->out.data, mem->out.shape[0], mem->out.shape[1]);
        #endif

        //residual connect
        profile_run(&profile_stats, PROFILE_ADD, [&](){
            add_inplace(&mem->in,&mem->out);
        });

    }

    #if defined(DEBUG)
    snprintf(logit_filename, sizeof(logit_filename), "./cppbin/output.bin");
    save_logits(logit_filename, (float*)mem->in.data, mem->in.shape[0], mem->in.shape[1]);
    #endif

    //result norm
    profile_run(&profile_stats, PROFILE_RMS_NORM, [&](){
        rms_norm(&mem->in,
                 &model_weights->output_norm,
                 nullptr,
                 model_config->rms_norm_eps);
    });

    #if defined(DEBUG)
    snprintf(logit_filename, sizeof(logit_filename), "./cppbin/output_norm.bin");
    save_logits(logit_filename, (float*)mem->in.data, mem->in.shape[0], mem->in.shape[1]);
    #endif

    //final logits, we only get the last token output
    mem->reset();
    mem->up.data = mem->allocate(1 * model_config->vocab_size * sizeof(float));
    mem->up.copy_meta(&mem->in);
    mem->up.shape[0] = 1;
    mem->up.shape[1] = model_config->vocab_size;

    Tensor last = mem->in.peek_at(mem->in.shape[0]-1);
    profile_run(&profile_stats, PROFILE_MATMUL_LM_HEAD, [&](){
        matmul( &last,
                &model_weights->lm_heads,
                &mem->up);
    });
    
    #if defined(DEBUG)
    snprintf(logit_filename, sizeof(logit_filename), "./cppbin/final_logits.bin");
    save_logits(logit_filename, (float*)mem->up.data, mem->up.shape[0], mem->up.shape[1]);
    #endif

    /*
    mem->reset();
    mem->up.data = mem->allocate(seq_len_inprocess * model_config->vocab_size * sizeof(float));
    mem->up.copy_meta(&mem->in);
    mem->up.shape[0] = seq_len_inprocess;
    mem->up.shape[1] = model_config->vocab_size;

    matmul( &mem->in,
            &model_weights->lm_heads,
            &mem->up);
    snprintf(logit_filename, sizeof(logit_filename), "./cppbin/final_logits.bin");
    //save_logits(logit_filename, (float*)mem->out.data, mem->out.shape[0], mem->out.shape[1]);
    save_logits(logit_filename, (float*)mem->up.data, mem->up.shape[0], mem->up.shape[1]);
    
    */

    mem->seq_len_processed += seq_len_inprocess;

    return &mem->up;
}

