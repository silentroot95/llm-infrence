
#include <assert.h>
#include <math.h>
#include <immintrin.h>

#include "tensor.h"



#if (defined(__AVX2__) && defined(__FMA__))
static inline float hsum256_ps(__m256 v) {
    __m128 vlow  = _mm256_castps256_ps128(v);
    __m128 vhigh = _mm256_extractf128_ps(v, 1);
    vlow = _mm_add_ps(vlow, vhigh);

    __m128 shuf = _mm_movehdup_ps(vlow);
    __m128 sums = _mm_add_ps(vlow, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);

    return _mm_cvtss_f32(sums);
}

static inline __m256 load_bf16_8_as_f32(const uint16_t* src) {
    __m128i bf16 = _mm_loadu_si128((const __m128i*)src);
    __m256i i32 = _mm256_cvtepu16_epi32(bf16);
    i32 = _mm256_slli_epi32(i32, 16);
    return _mm256_castsi256_ps(i32);
}

static inline void store_f32_tensor_to_bf16(const Tensor* src, Tensor* dst) {
    int num = src->nelements();
    float* src_data = (float*)src->data;
    uint16_t* dst_data = (uint16_t*)dst->data;

    int i = 0;
    for(; i + 7 < num; i += 8) {
        __m256 f32 = _mm256_loadu_ps(src_data + i);
        __m256i bits = _mm256_castps_si256(f32);
        __m256i hi16 = _mm256_srli_epi32(bits, 16);

        __m128i lo = _mm256_castsi256_si128(hi16);
        __m128i hi = _mm256_extracti128_si256(hi16, 1);
        __m128i packed = _mm_packus_epi32(lo, hi);
        _mm_storeu_si128((__m128i*)(dst_data + i), packed);
    }

    for(; i < num; ++i) {
        dst_data[i] = f32_to_bf16(src_data[i]);
    }
}

static inline void simd_gemv_f32(const float* x, const float* weight, float* out, int hidden_dim, int out_dim) {
    const int block_count = out_dim / 4;

    #pragma omp parallel for if(out_dim >= 1024) schedule(static)
    for(int bi = 0; bi < block_count; ++bi) {
        const int oi = bi * 4;
        const float* w0 = weight + (oi + 0) * hidden_dim;
        const float* w1 = weight + (oi + 1) * hidden_dim;
        const float* w2 = weight + (oi + 2) * hidden_dim;
        const float* w3 = weight + (oi + 3) * hidden_dim;

        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        __m256 acc2 = _mm256_setzero_ps();
        __m256 acc3 = _mm256_setzero_ps();

        int c = 0;
        for(; c + 7 < hidden_dim; c += 8) {
            __m256 xv = _mm256_loadu_ps(x + c);
            acc0 = _mm256_fmadd_ps(xv, _mm256_loadu_ps(w0 + c), acc0);
            acc1 = _mm256_fmadd_ps(xv, _mm256_loadu_ps(w1 + c), acc1);
            acc2 = _mm256_fmadd_ps(xv, _mm256_loadu_ps(w2 + c), acc2);
            acc3 = _mm256_fmadd_ps(xv, _mm256_loadu_ps(w3 + c), acc3);
        }

        float s0 = hsum256_ps(acc0);
        float s1 = hsum256_ps(acc1);
        float s2 = hsum256_ps(acc2);
        float s3 = hsum256_ps(acc3);
        for(; c < hidden_dim; ++c) {
            float xv = x[c];
            s0 += xv * w0[c];
            s1 += xv * w1[c];
            s2 += xv * w2[c];
            s3 += xv * w3[c];
        }

        out[oi + 0] = s0;
        out[oi + 1] = s1;
        out[oi + 2] = s2;
        out[oi + 3] = s3;
    }

    for(int oi = block_count * 4; oi < out_dim; ++oi) {
        const float* w = weight + oi * hidden_dim;
        __m256 acc = _mm256_setzero_ps();
        int c = 0;
        for(; c + 7 < hidden_dim; c += 8) {
            __m256 xv = _mm256_loadu_ps(x + c);
            __m256 wv = _mm256_loadu_ps(w + c);
            acc = _mm256_fmadd_ps(xv, wv, acc);
        }

        float s = hsum256_ps(acc);
        for(; c < hidden_dim; ++c) {
            s += x[c] * w[c];
        }
        out[oi] = s;
    }
}
#endif


void softmax(float* x, int len, float t=1.0) {
    float maxval = x[0];
    for(int i=1;i<len;++i) {
        if(x[i] > maxval) {
            maxval = x[i];
        }
    }

    float sum=0.f;
    for(int i=0;i<len;++i) {
        x[i] = expf((x[i]-maxval) / t);
        sum += x[i];
    }

    for(int i=0;i<len;++i) {
        x[i] /= sum;
    }
}


void embed(const Tensor* tokens, const Tensor* embed_weight, Tensor* out) {
    float* o = (float*)out->data;
    int n = tokens->shape[0];
    int embed_size = embed_weight->shape[1];
    int* ids = (int*)tokens->data;

    if(embed_weight->data_type == TENSOR_DATA_TYPE_BF16) {
        const uint16_t* embed = (const uint16_t*)embed_weight->data;
        for(int r=0;r<n;++r) {
            float* ro = o + r*embed_size;
            const uint16_t* eo = embed + ids[r]*embed_size;
            for(int i=0;i<embed_size;++i) {
                ro[i] = bf16_to_f32(eo[i]);
            }
        }
    }
    //default is F32
    else {
        const float* embed = (const float*)embed_weight->data;
        for(int i=0;i<n;++i) {
            memcpy(o+i*embed_size,embed + ids[i]*embed_size, sizeof(float)*embed_size);
        }
    }
}


void rms_norm(const Tensor* tb, const Tensor* norm_weight, Tensor* out, float eps) {
    int d = norm_weight->shape[0];
    assert(tb->shape[tb->dim - 1] == d);

    int rows = tb->nelements() / d;

    const float* weight = (const float*)norm_weight->data;
    const float* data = (const float*)tb->data;
    float* out_data = nullptr;

    // if out is nullptr, do inplace norm, otherwise write to out
    if(out == nullptr) {
        out_data = (float*)tb->data;
    }
    else {
        out_data = (float*)out->data;
    }

    for(size_t i=0; i<rows; ++i) {
        const float* x = data + i*d;
        float* o = out_data + i*d;
        float ss = 0.0f;
        
        #if defined(__AVX2__) && defined(__FMA__)
            
            __m256 sum_vec = _mm256_setzero_ps();
            for (size_t j = 0; j <= d - 8; j += 8) {
                __m256 v = _mm256_loadu_ps(x + j);
                sum_vec = _mm256_fmadd_ps(v, v, sum_vec);
            }

            ss = hsum256_ps(sum_vec);

            // handle tail when d is not divisible by 8
            for (size_t j = (d/8)*8; j < d; j++) {
                ss += x[j] * x[j];
            }

            float inv_rms = 1.0f / sqrtf(ss / d + eps);
            //printf("inv_rms: %f\n", inv_rms);
            assert(inv_rms > 0.0f);
            __m256 v_inv_rms = _mm256_set1_ps(inv_rms);
            
            for (size_t j = 0; j <= d - 8; j += 8) {
                __m256 v_x = _mm256_loadu_ps(x + j);
                __m256 v_w = _mm256_loadu_ps(weight + j);
                __m256 v_res = _mm256_mul_ps(_mm256_mul_ps(v_x, v_inv_rms), v_w);
                _mm256_storeu_ps(o + j, v_res);
            }
            for (size_t j=(d/8)*8;j<d;++j) {
                o[j] = x[j] * inv_rms * weight[j];
            }
        #else
        for(int j=0;j<d;++j) {
            ss += x[j] * x[j];
        }

        float inv_rms = 1.0f / sqrtf(ss/d + eps);
        assert(inv_rms > 0.0f);
        for (size_t j=0;j<d;++j) {
            o[j] = x[j] * inv_rms * weight[j];
        }
        #endif
    }
}


//seq_len_processed is length that has processed
// tb shape is [seq_len, head, head_dim]
void rope(Tensor* tb, float theta_base, int seq_len_processed) {
    int seq_len   = tb->shape[0];
    int num_heads = tb->shape[1];
    int head_dim  = tb->shape[2];
    int half      = head_dim / 2;

    float* data = (float*)tb->data;

    //#pragma omp parallel for if (seq_len >= 1024) schedule(static)
    for (int r = 0; r < seq_len; ++r) {
        int pos = seq_len_processed + r;

        for (int h = 0; h < num_heads; ++h) {
            float* head_data = data + (r * num_heads + h) * head_dim;

            for (int i = 0; i < half; ++i) {
                double exponent = (double)(2 * i) / head_dim;
                double inv_freq = pow((double)theta_base, -exponent);
                double theta = pos * inv_freq;

                double cos_t = cos(theta);
                double sin_t = sin(theta);

                double a = head_data[i];
                double b = head_data[i + half];

                head_data[i]        = (float)(a * cos_t - b * sin_t);
                head_data[i + half] = (float)(a * sin_t + b * cos_t);
            }
        }
    }
}

void attention(Tensor* q,Tensor* k, Tensor* v, int seq_len_processed, float* attn) {
    /*
      q shape is [sequence,q_head,head_dim]
      kv shape is [sequence,kv_head,head_dim]
    */
    assert(q->dim == 3);
    int q_head = q->shape[1];
    int head_dim = q->shape[2];
    int q_len = q->shape[0];

    int kv_head = k->shape[1];

    int group_size = q_head / kv_head;
    float scale = 1.0f / sqrtf(head_dim);

    float* qdata = (float*)q->data;
    const float* kdata = (const float*)k->data;
    const float* vdata = (const float*)v->data;

    if (q_len == 1) {
        const int total_kv = seq_len_processed + 1;
        #pragma omp parallel for schedule(static)
        for(int h=0; h<q_head; ++h) {
            int kv_selected = h / group_size;
            float* q_data = qdata + h * head_dim;
            float* attn_data = attn + h * total_kv;

            int kv_index = 0;
            for(; kv_index + 3 < total_kv; kv_index += 4) {
                int base0 = (kv_index + 0) * head_dim * kv_head + kv_selected * head_dim;
                int base1 = (kv_index + 1) * head_dim * kv_head + kv_selected * head_dim;
                int base2 = (kv_index + 2) * head_dim * kv_head + kv_selected * head_dim;
                int base3 = (kv_index + 3) * head_dim * kv_head + kv_selected * head_dim;
                const float* k_data0 = kdata + base0;
                const float* k_data1 = kdata + base1;
                const float* k_data2 = kdata + base2;
                const float* k_data3 = kdata + base3;

                #if defined(__AVX2__) && defined(__FMA__)
                __m256 acc0 = _mm256_setzero_ps();
                __m256 acc1 = _mm256_setzero_ps();
                __m256 acc2 = _mm256_setzero_ps();
                __m256 acc3 = _mm256_setzero_ps();
                int i = 0;
                for(; i + 7 < head_dim; i += 8) {
                    __m256 qv = _mm256_loadu_ps(q_data + i);
                    __m256 k0 = _mm256_loadu_ps(k_data0 + i);
                    __m256 k1 = _mm256_loadu_ps(k_data1 + i);
                    __m256 k2 = _mm256_loadu_ps(k_data2 + i);
                    __m256 k3 = _mm256_loadu_ps(k_data3 + i);
                    acc0 = _mm256_fmadd_ps(qv, k0, acc0);
                    acc1 = _mm256_fmadd_ps(qv, k1, acc1);
                    acc2 = _mm256_fmadd_ps(qv, k2, acc2);
                    acc3 = _mm256_fmadd_ps(qv, k3, acc3);
                }

                float s0 = hsum256_ps(acc0);
                float s1 = hsum256_ps(acc1);
                float s2 = hsum256_ps(acc2);
                float s3 = hsum256_ps(acc3);
                for(; i < head_dim; ++i) {
                    float qv = q_data[i];
                    s0 += qv * k_data0[i];
                    s1 += qv * k_data1[i];
                    s2 += qv * k_data2[i];
                    s3 += qv * k_data3[i];
                }
                #else
                float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
                for(int i = 0; i < head_dim; ++i) {
                    float qv = q_data[i];
                    s0 += qv * k_data0[i];
                    s1 += qv * k_data1[i];
                    s2 += qv * k_data2[i];
                    s3 += qv * k_data3[i];
                }
                #endif
                attn_data[kv_index + 0] = s0 * scale;
                attn_data[kv_index + 1] = s1 * scale;
                attn_data[kv_index + 2] = s2 * scale;
                attn_data[kv_index + 3] = s3 * scale;
            }

            for(; kv_index < total_kv; ++kv_index) {
                int base = kv_index * head_dim * kv_head + kv_selected * head_dim;
                const float* k_data = kdata + base;

                #if defined(__AVX2__) && defined(__FMA__)
                __m256 acc = _mm256_setzero_ps();
                int i = 0;
                for(; i + 7 < head_dim; i += 8) {
                    __m256 qv = _mm256_loadu_ps(q_data + i);
                    __m256 kv = _mm256_loadu_ps(k_data + i);
                    acc = _mm256_fmadd_ps(qv, kv, acc);
                }
                float s = hsum256_ps(acc);
                for(; i < head_dim; ++i) {
                    s += q_data[i] * k_data[i];
                }
                #else
                float s = 0.0f;
                for(int i = 0; i < head_dim; ++i) {
                    s += q_data[i] *  k_data[i];
                }
                #endif
                attn_data[kv_index] = s * scale;
            }

            softmax(attn_data, total_kv);
            memset(q_data, 0, sizeof(float) * head_dim);

            for(int kv_index = 0; kv_index < total_kv; ++kv_index) {
                int base = kv_index * head_dim * kv_head + kv_selected * head_dim;
                const float* v_data = vdata + base;

                #if defined(__AVX2__) && defined(__FMA__)
                __m256 attn_vec = _mm256_set1_ps(attn_data[kv_index]);
                int i = 0;
                for(; i + 7 < head_dim; i += 8) {
                    __m256 outv = _mm256_loadu_ps(q_data + i);
                    __m256 vv =  _mm256_loadu_ps(v_data + i);
                    outv = _mm256_fmadd_ps(attn_vec, vv, outv);
                    _mm256_storeu_ps(q_data + i, outv);
                }
                for(; i < head_dim; ++i) {
                    q_data[i] += attn_data[kv_index] * v_data[i];
                }
                #else
                for(int i = 0; i < head_dim; ++i) {
                    q_data[i] += attn_data[kv_index] *  v_data[i];
                }
                #endif
            }
        }
        return;
    }

    #pragma omp parallel for schedule(static)
    for(int h=0;h<q_head;++h) {
        int kv_selected = h / group_size;
        for(int q_index=0;q_index<q_len;++q_index) {
            float* q_data = qdata + q_index * head_dim * q_head + h * head_dim;

            float* attn_data = attn + h * (q_len + seq_len_processed); 
            
            #if defined(__AVX2__) && defined(__FMA__)
            int kv_index = 0;
            // q can only attention the kv before current q_index
            for(; kv_index + 3 <=(q_index + seq_len_processed); kv_index+=4) {
                const float* k_data0 = kdata + (kv_index + 0) * head_dim * kv_head + kv_selected * head_dim;
                const float* k_data1 = kdata + (kv_index + 1) * head_dim * kv_head + kv_selected * head_dim;
                const float* k_data2 = kdata + (kv_index + 2) * head_dim * kv_head + kv_selected * head_dim;
                const float* k_data3 = kdata + (kv_index + 3) * head_dim * kv_head + kv_selected * head_dim;

                __m256 acc0 = _mm256_setzero_ps();
                __m256 acc1 = _mm256_setzero_ps();
                __m256 acc2 = _mm256_setzero_ps();
                __m256 acc3 = _mm256_setzero_ps();

                int i = 0;
                for(; i + 7 < head_dim; i += 8) {
                    __m256 qv = _mm256_loadu_ps(q_data + i);
                    __m256 k0 = _mm256_loadu_ps(k_data0 + i);
                    __m256 k1 = _mm256_loadu_ps(k_data1 + i);
                    __m256 k2 = _mm256_loadu_ps(k_data2 + i);
                    __m256 k3 = _mm256_loadu_ps(k_data3 + i);

                    acc0 = _mm256_fmadd_ps(qv, k0, acc0);
                    acc1 = _mm256_fmadd_ps(qv, k1, acc1);
                    acc2 = _mm256_fmadd_ps(qv, k2, acc2);
                    acc3 = _mm256_fmadd_ps(qv, k3, acc3);
                }

                float s0 = hsum256_ps(acc0);
                float s1 = hsum256_ps(acc1);
                float s2 = hsum256_ps(acc2);
                float s3 = hsum256_ps(acc3);

                for(; i < head_dim; ++i) {
                    float qv = q_data[i];
                    s0 += qv * k_data0[i];
                    s1 += qv * k_data1[i];
                    s2 += qv * k_data2[i];
                    s3 += qv * k_data3[i];
                }
                attn_data[kv_index+0] = s0 * scale;
                attn_data[kv_index+1] = s1 * scale;
                attn_data[kv_index+2] = s2 * scale;
                attn_data[kv_index+3] = s3 * scale;
            }

            for(; kv_index <= (q_index + seq_len_processed); ++kv_index) {
                const float* k_data = kdata + kv_index * head_dim * kv_head + kv_selected * head_dim;

                __m256 acc = _mm256_setzero_ps();
                int i = 0;  
                for(; i + 7 < head_dim; i += 8) {
                    __m256 qv = _mm256_loadu_ps(q_data + i);
                    __m256 kv = _mm256_loadu_ps(k_data + i);
                    acc = _mm256_fmadd_ps(qv, kv, acc);
                }
                float s = hsum256_ps(acc);
                for(; i < head_dim; ++i) {
                    s += q_data[i] *  k_data[i];
                }
                attn_data[kv_index] = s * scale;
            }

            #else
            for(int kv_index=0; kv_index<=(q_index + seq_len_processed); ++kv_index) {
                const float* k_data = kdata + kv_index * head_dim * kv_head + kv_selected * head_dim;
                float sum = 0.0f;
                for(int i=0;i<head_dim;++i) {
                    sum += q_data[i] * k_data[i];
                }
                attn_data[kv_index] = sum * scale;
            }
            #endif

            softmax(attn_data, q_index+1+seq_len_processed);
            memset(q_data,0,sizeof(float) * head_dim);

            for(int kv_index=0; kv_index<=(q_index + seq_len_processed); ++kv_index) {
                int base = kv_index * head_dim * kv_head + kv_selected * head_dim;
                const float* v_data = vdata + base;
                #if defined(__AVX2__) && defined(__FMA__)
                __m256 attn_vec = _mm256_set1_ps(attn_data[kv_index]);
                int i = 0;
                for(; i + 7 < head_dim; i += 8) {
                    __m256 outv = _mm256_loadu_ps(q_data + i);
                    __m256 vv =  _mm256_loadu_ps(v_data + i);
                    outv = _mm256_fmadd_ps(attn_vec, vv, outv);
                    _mm256_storeu_ps(q_data + i, outv);
                }
                for(; i < head_dim; ++i) {
                    q_data[i] += attn_data[kv_index] * v_data[i];
                }
                #else
                for(int i=0;i<head_dim;++i) {
                    q_data[i] += attn_data[kv_index] * v_data[i];
                }
                #endif
            }
        }
    }
    // now q is softmaxt(q@k.T/sqrt(d))@V shape [sequence,q_heads,head_size]
}


/*
  Tensor matrix multiply like x@Q
  tensor x shape: [S,H]
  tensor Q shape: [out_dim,H]
*/
void matmul(const Tensor* tb,const Tensor* weight , Tensor* out) {
    //tensor matmul at the last two dimension
    int hidden_size = tb->shape[tb->dim-1];
    assert(tb->dim == weight->dim);
    assert(tb->shape[tb->dim-1] == weight->shape[weight->dim-1]);
    assert(hidden_size == weight->shape[weight->dim-1]);

    int out_size = weight->shape[weight->dim-2];
    int row = tb->nelements() / hidden_size;
    int seq_len = tb->shape[0];

    const float* data = (const float*)tb->data;
    const float* weight_data = (const float*)weight->data;
    float* out_data = (float*)out->data;

    if(seq_len == 1) {
        #if defined(__AVX2__) && defined(__FMA__)
        simd_gemv_f32(data, weight_data, out_data, hidden_size, out_size);
        #else
        for(int k=0; k<out_size; ++k) {
            float* w = (float*)weight->data + k * hidden_size;
            float s = 0.0f;
            for(int i=0; i<hidden_size; ++i) {
                s += data[i] * w[i];
            }
            out_data[k] = s;
        }
        #endif
        return;
    }
    
    #pragma omp parallel for if(row >= 64) schedule(static)
    for(int r=0; r<row; ++r) {
        float* o =  out_data + r*out_size;
        const float* x = data + r*hidden_size;

        #if defined(__AVX2__) && defined(__FMA__)
        int k = 0;
        for(; k + 3 < out_size; k += 4) {
            const float* w0 = weight_data + (k + 0) * hidden_size;
            const float* w1 = weight_data + (k + 1) * hidden_size;
            const float* w2 = weight_data + (k + 2) * hidden_size;
            const float* w3 = weight_data + (k + 3) * hidden_size;

            __m256 acc0 = _mm256_setzero_ps();
            __m256 acc1 = _mm256_setzero_ps();
            __m256 acc2 = _mm256_setzero_ps();
            __m256 acc3 = _mm256_setzero_ps();

            int i = 0;
            for(; i + 7 < hidden_size; i += 8) {
                __m256 xv = _mm256_loadu_ps(x + i);
                acc0 = _mm256_fmadd_ps(xv, _mm256_loadu_ps(w0 + i), acc0);
                acc1 = _mm256_fmadd_ps(xv, _mm256_loadu_ps(w1 + i), acc1);
                acc2 = _mm256_fmadd_ps(xv, _mm256_loadu_ps(w2 + i), acc2);
                acc3 = _mm256_fmadd_ps(xv, _mm256_loadu_ps(w3 + i), acc3);
            }

            float s0 = hsum256_ps(acc0);
            float s1 = hsum256_ps(acc1);
            float s2 = hsum256_ps(acc2);
            float s3 = hsum256_ps(acc3);

            // Tail handle when hidden_size is not divisible by 8.
            for(; i < hidden_size; ++i) {
                float xv = x[i];
                s0 += xv * w0[i];
                s1 += xv * w1[i];
                s2 += xv * w2[i];
                s3 += xv * w3[i];
            }

            o[k + 0] = s0;
            o[k + 1] = s1;
            o[k + 2] = s2;
            o[k + 3] = s3;
        }

        // Handle remaining outputs that don't fill a 4-column block.
        for(; k < out_size; ++k) {
            const float* w = weight_data + k * hidden_size;
            __m256 acc = _mm256_setzero_ps();
            int i = 0;
            for(; i + 7 < hidden_size; i += 8) {
                __m256 xv = _mm256_loadu_ps(x + i);
                __m256 wv = _mm256_loadu_ps(w + i);
                acc = _mm256_fmadd_ps(xv, wv, acc);
            }

            float s = hsum256_ps(acc);
            for(; i < hidden_size; ++i) {
                s += x[i] * w[i];
            }
            o[k] = s;
        }
        #else
        for(int k=0; k<out_size; ++k) {
            const float* w = (const float*)weight->data + k*hidden_size;
            float ss = 0.0;
            for(int i=0; i<hidden_size; ++i) {
                ss += x[i] * w[i];
            }
           
            o[k] = ss;
        }
        #endif
    }
}

static inline float stable_sigmoid(float x) {
    if(x > 0) {
        float z = 1.0f / (1.0f + expf(-x));
        return z;
    }
    else {
        float z = expf(x);
        return  z / (1.0f + z);
    }
}

void mlp(Tensor* up, Tensor* gate) {
    const float* g_data = (const float*)gate->data;
    float* u_data = (float*)up->data; 
    int n = up->nelements();
    for(int i=0;i<n; ++i) {
        float silu = g_data[i] * stable_sigmoid(g_data[i]);
        u_data[i] *= silu;
    }
}

inline float silu(float x) {
    return x * stable_sigmoid(x);
}

void up_gate_mlp(const Tensor* in,
                 const Tensor* up_weight,
                 const Tensor* gate_weight,
                 Tensor* up) {
    
    int out_size = up_weight->shape[up_weight->dim - 2];
    int hidden_size = up_weight->shape[up_weight->dim - 1];
    int seq_len = in->shape[0];

    const float* data = (const float*)in->data;
    const float* up_weight_data = (const float*)up_weight->data;
    const float* gate_weight_data = (const float*)gate_weight->data;
    float* out_data = (float*)up->data; 

    int main_k_end = (out_size / 4) * 4;
    // decode
    if (seq_len == 1) {
        const float* x = data;
        
        #if defined(__AVX2__) && defined(__FMA__)
        #pragma omp parallel for if(out_size >= 512) schedule(static)
        for(int k = 0; k < out_size - 3; k += 4) {
            const float* w_u0 = up_weight_data + (k + 0) * hidden_size;
            const float* w_u1 = up_weight_data + (k + 1) * hidden_size;
            const float* w_u2 = up_weight_data + (k + 2) * hidden_size;
            const float* w_u3 = up_weight_data + (k + 3) * hidden_size;

            const float* w_g0 = gate_weight_data + (k + 0) * hidden_size;
            const float* w_g1 = gate_weight_data + (k + 1) * hidden_size;
            const float* w_g2 = gate_weight_data + (k + 2) * hidden_size;
            const float* w_g3 = gate_weight_data + (k + 3) * hidden_size;

            __m256 acc_u0 = _mm256_setzero_ps();
            __m256 acc_u1 = _mm256_setzero_ps();
            __m256 acc_u2 = _mm256_setzero_ps();
            __m256 acc_u3 = _mm256_setzero_ps();

            __m256 acc_g0 = _mm256_setzero_ps();
            __m256 acc_g1 = _mm256_setzero_ps();
            __m256 acc_g2 = _mm256_setzero_ps();
            __m256 acc_g3 = _mm256_setzero_ps();

            int i = 0;
            for(; i + 7 < hidden_size; i += 8) {
                __m256 xv = _mm256_loadu_ps(x + i);
                
                //  up
                acc_u0 = _mm256_fmadd_ps(xv, _mm256_loadu_ps(w_u0 + i), acc_u0);
                acc_u1 = _mm256_fmadd_ps(xv, _mm256_loadu_ps(w_u1 + i), acc_u1);
                acc_u2 = _mm256_fmadd_ps(xv, _mm256_loadu_ps(w_u2 + i), acc_u2);
                acc_u3 = _mm256_fmadd_ps(xv, _mm256_loadu_ps(w_u3 + i), acc_u3);

                //  gate
                acc_g0 = _mm256_fmadd_ps(xv, _mm256_loadu_ps(w_g0 + i), acc_g0);
                acc_g1 = _mm256_fmadd_ps(xv, _mm256_loadu_ps(w_g1 + i), acc_g1);
                acc_g2 = _mm256_fmadd_ps(xv, _mm256_loadu_ps(w_g2 + i), acc_g2);
                acc_g3 = _mm256_fmadd_ps(xv, _mm256_loadu_ps(w_g3 + i), acc_g3);
            }

            float s_u0 = hsum256_ps(acc_u0);
            float s_u1 = hsum256_ps(acc_u1);
            float s_u2 = hsum256_ps(acc_u2);
            float s_u3 = hsum256_ps(acc_u3);

            float s_g0 = hsum256_ps(acc_g0);
            float s_g1 = hsum256_ps(acc_g1);
            float s_g2 = hsum256_ps(acc_g2);
            float s_g3 = hsum256_ps(acc_g3);

           
            for(; i < hidden_size; ++i) {
                float xv = x[i];
                s_u0 += xv * w_u0[i];
                s_u1 += xv * w_u1[i];
                s_u2 += xv * w_u2[i];
                s_u3 += xv * w_u3[i];

                s_g0 += xv * w_g0[i];
                s_g1 += xv * w_g1[i];
                s_g2 += xv * w_g2[i];
                s_g3 += xv * w_g3[i];
            }

            // SiLU(gate) * up
            out_data[k + 0] = silu(s_g0) * s_u0;
            out_data[k + 1] = silu(s_g1) * s_u1;
            out_data[k + 2] = silu(s_g2) * s_u2;
            out_data[k + 3] = silu(s_g3) * s_u3;
        }

        for(int k = main_k_end; k < out_size; ++k) {
            const float* w_u = up_weight_data + k * hidden_size;
            const float* w_g = gate_weight_data + k * hidden_size;
            
            __m256 acc_u = _mm256_setzero_ps();
            __m256 acc_g = _mm256_setzero_ps();
            
            int i = 0;
            for(; i + 7 < hidden_size; i += 8) {
                __m256 xv = _mm256_loadu_ps(x + i);
                acc_u = _mm256_fmadd_ps(xv, _mm256_loadu_ps(w_u + i), acc_u);
                acc_g = _mm256_fmadd_ps(xv, _mm256_loadu_ps(w_g + i), acc_g);
            }

            float s_u = hsum256_ps(acc_u);
            float s_g = hsum256_ps(acc_g);
            for(; i < hidden_size; ++i) {
                s_u += x[i] * w_u[i];
                s_g += x[i] * w_g[i];
            }
            out_data[k] = silu(s_g) * s_u;
        }

        #else
        // Plain C++ fallback for seq_len == 1
        #pragma omp parallel for if(out_size >= 1024) schedule(static)
        for(int k=0; k<out_size; ++k) {
            const float* w_u = up_weight_data + k * hidden_size;
            const float* w_g = gate_weight_data + k * hidden_size;
            float sum_u = 0.0f;
            float sum_g = 0.0f;
            
            for(int i=0; i<hidden_size; ++i) {
                float xv = x[i];
                sum_u += xv * w_u[i];
                sum_g += xv * w_g[i];
            }
            out_data[k] = silu(sum_g) * sum_u;
        }
        #endif
        return;
    }

    // prefill
    #pragma omp parallel for schedule(static)
    for(int r = 0; r < seq_len; ++r) {
        float* o = out_data + r * out_size;
        const float* x = data + r * hidden_size;

        #if defined(__AVX2__) && defined(__FMA__)
        for(int k = 0; k < out_size-3; k += 4) {
            const float* w_u0 = up_weight_data + (k + 0) * hidden_size;
            const float* w_u1 = up_weight_data + (k + 1) * hidden_size;
            const float* w_u2 = up_weight_data + (k + 2) * hidden_size;
            const float* w_u3 = up_weight_data + (k + 3) * hidden_size;

            const float* w_g0 = gate_weight_data + (k + 0) * hidden_size;
            const float* w_g1 = gate_weight_data + (k + 1) * hidden_size;
            const float* w_g2 = gate_weight_data + (k + 2) * hidden_size;
            const float* w_g3 = gate_weight_data + (k + 3) * hidden_size;

            __m256 acc_u0 = _mm256_setzero_ps();
            __m256 acc_u1 = _mm256_setzero_ps();
            __m256 acc_u2 = _mm256_setzero_ps();
            __m256 acc_u3 = _mm256_setzero_ps();

            __m256 acc_g0 = _mm256_setzero_ps();
            __m256 acc_g1 = _mm256_setzero_ps();
            __m256 acc_g2 = _mm256_setzero_ps();
            __m256 acc_g3 = _mm256_setzero_ps();

            int i = 0;
            for(; i + 7 < hidden_size; i += 8) {
                __m256 xv = _mm256_loadu_ps(x + i);
                
                acc_u0 = _mm256_fmadd_ps(xv, _mm256_loadu_ps(w_u0 + i), acc_u0);
                acc_u1 = _mm256_fmadd_ps(xv, _mm256_loadu_ps(w_u1 + i), acc_u1);
                acc_u2 = _mm256_fmadd_ps(xv, _mm256_loadu_ps(w_u2 + i), acc_u2);
                acc_u3 = _mm256_fmadd_ps(xv, _mm256_loadu_ps(w_u3 + i), acc_u3);

                acc_g0 = _mm256_fmadd_ps(xv, _mm256_loadu_ps(w_g0 + i), acc_g0);
                acc_g1 = _mm256_fmadd_ps(xv, _mm256_loadu_ps(w_g1 + i), acc_g1);
                acc_g2 = _mm256_fmadd_ps(xv, _mm256_loadu_ps(w_g2 + i), acc_g2);
                acc_g3 = _mm256_fmadd_ps(xv, _mm256_loadu_ps(w_g3 + i), acc_g3);
            }

            float s_u0 = hsum256_ps(acc_u0); 
            float s_u1 = hsum256_ps(acc_u1);
            float s_u2 = hsum256_ps(acc_u2); 
            float s_u3 = hsum256_ps(acc_u3);
            
            float s_g0 = hsum256_ps(acc_g0); 
            float s_g1 = hsum256_ps(acc_g1);
            float s_g2 = hsum256_ps(acc_g2); 
            float s_g3 = hsum256_ps(acc_g3);

            for(; i < hidden_size; ++i) {
                float xv = x[i];
                s_u0 += xv * w_u0[i]; s_u1 += xv * w_u1[i];
                s_u2 += xv * w_u2[i]; s_u3 += xv * w_u3[i];

                s_g0 += xv * w_g0[i]; s_g1 += xv * w_g1[i];
                s_g2 += xv * w_g2[i]; s_g3 += xv * w_g3[i];
            }

            o[k + 0] = silu(s_g0) * s_u0;
            o[k + 1] = silu(s_g1) * s_u1;
            o[k + 2] = silu(s_g2) * s_u2;
            o[k + 3] = silu(s_g3) * s_u3;
        }

        for(int k = main_k_end; k < out_size; ++k) {
            const float* w_u = up_weight_data + k * hidden_size;
            const float* w_g = gate_weight_data + k * hidden_size;
            __m256 acc_u = _mm256_setzero_ps();
            __m256 acc_g = _mm256_setzero_ps();
            int i = 0;
            for(; i + 7 < hidden_size; i += 8) {
                __m256 xv = _mm256_loadu_ps(x + i);
                acc_u = _mm256_fmadd_ps(xv, _mm256_loadu_ps(w_u + i), acc_u);
                acc_g = _mm256_fmadd_ps(xv, _mm256_loadu_ps(w_g + i), acc_g);
            }
            float s_u = hsum256_ps(acc_u);
            float s_g = hsum256_ps(acc_g);
            for(; i < hidden_size; ++i) {
                s_u += x[i] * w_u[i];
                s_g += x[i] * w_g[i];
            }
            o[k] = silu(s_g) * s_u;
        }

        #else
        for(int k=0; k<out_size; ++k) {
            const float* w_u = up_weight_data + k * hidden_size;
            const float* w_g = gate_weight_data + k * hidden_size;
            float sum_u = 0.0f;
            float sum_g = 0.0f;
            for(int i=0; i<hidden_size; ++i) {
                float xv = x[i];
                sum_u += xv * w_u[i];
                sum_g += xv * w_g[i];
            }
            o[k] = silu(sum_g) * sum_u;
        }
        #endif
    }
}

void matmul_w8a32(const Tensor* in, const QuantizedWeightINT8* q_weight, Tensor* out) {
    int out_size = q_weight->shape[q_weight->dim - 2];
    int hidden_size = q_weight->shape[q_weight->dim - 1];

    const float* x = (const float*)in->data;
    float* out_data = (float*)out->data;
    const int8_t* q_weight_data = q_weight->q_weight;
    const float* scales = q_weight->scales;

    #pragma omp parallel for schedule(static)
    for (int k = 0; k < out_size; ++k) {
        const int8_t* w_row = q_weight_data + k * hidden_size;
        float row_scale = scales[k];
        
        #if defined(__AVX2__) && defined(__FMA__)
        __m256 acc = _mm256_setzero_ps();
        int i = 0;
        
        for (; i + 7 < hidden_size; i += 8) {
            // load x
            __m256 xv = _mm256_loadu_ps(x + i);
            
            //loas q_weight
            __m128i w8 = _mm_loadl_epi64((const __m128i*)(w_row + i));
            // convert INT8 to INT32
            __m256i w32 = _mm256_cvtepi8_epi32(w8);
            // convert INT32 to FP32
            __m256 wf = _mm256_cvtepi32_ps(w32);
            
            acc = _mm256_fmadd_ps(xv, wf, acc);
        }
        
        float sum = hsum256_ps(acc);
        
        for (; i < hidden_size; ++i) {
            sum += x[i] * (float)w_row[i];
        }
        
        out_data[k] = sum * row_scale;
        
        #else
        float sum = 0.0f;
        for (int i = 0; i < hidden_size; ++i) {
            sum += x[i] * (float)w_row[i];
        }
        out_data[k] = sum * row_scale;
        #endif
    }
}