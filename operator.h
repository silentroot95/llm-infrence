#pragma once

struct Tensor;

void softmax(float* x, int pos, float t=1.0);

void embed(const Tensor* tokens, const Tensor* embed_weight, Tensor* out);

void rms_norm(const Tensor* tb, const Tensor* norm_weight, Tensor* out, float eps);
     
//seq_len_processed is length that has processed
// tb shape is [seq_len, head, head_dim]
void rope(Tensor* tb, float theta_base, int seq_len_processed);

void attention(Tensor* q,Tensor* k, Tensor* v, int seq_len_processed, float* attn);

/*
  Tensor matrix multiply like x@Q
  tensor x shape: [S,H]
  tensor Q shape: [out_dim,H]
*/
void matmul(const Tensor* tb,const Tensor* weight , Tensor* out);

void sigmoid(Tensor* x, Tensor* out);

void silu(Tensor* x);

void mlp(Tensor* up, Tensor* gate);

inline void add_inplace(const Tensor* in,const Tensor* param) {
    // assert(in->nelements() == param->nelements());
    float* in_data = (float*)in->data;
    const float* param_data = (const float*)param->data;
    for(int i=0; i<in->nelements(); ++i) {
        in_data[i] += param_data[i];
    }
}

inline void multiply_inplace(Tensor* a, const Tensor* b) {
    int n = a->nelements();
    float* a_data = (float*)a->data;
    const float* b_data = (const float*)b->data;
    for(int i=0; i<n; ++i) {
        a_data[i] *= b_data[i];
    }
}

inline int argmax(const Tensor* logits) {
    int n = logits->nelements();
    int max_id = 0;
    float* data = (float*)logits->data;
    float max_val = data[0];
    for(int i=1;i<n;++i) {
        if(data[i] > max_val) {
            max_val = data[i];
            max_id = i;
        }
    }
    return max_id;
}

