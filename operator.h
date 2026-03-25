#pragma once

#include <assert.h>
#include <math.h>
#include <immintrin.h>
#include <vector>

#include "tensor.h"



enum OPTYPE {
    OP_TENSOR_ADD,
    OP_TENSOR_EMBED,
    OP_TENSOR_NORM,
    OP_TENSOR_RESHAPE_3D,
    OP_TENSOR_ROPE,
    OP_TENSOR_MATMUL,
    OP_TENSOR_ATTENTION,
    OP_TENSOR_MLP
};


struct TOperator {
    TOperator() : out(nullptr) {}
    OPTYPE operator_type;
    const Tensor* params[8];
    Tensor* out;
    int32_t other_params[8];

    void forward() {}
};


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

    #if defined(__AVX2__)
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
    #else
    for (int i = 0; i < num; ++i) {
        dst_data[i] = f32_to_bf16(src_data[i]);
    }
    #endif
}

void softmax(float* x, int pos, float t=1.0) {
    float maxval = x[0];
    for(int i=1;i<pos;++i) {
        if(x[i] > maxval) {
            maxval = x[i];
        }
    }

    float sum=0.f;
    for(int i=0;i<pos;++i) {
        x[i] = expf((x[i]-maxval) / t);
        sum += x[i];
    }

    for(int i=0;i<pos;++i) {
        x[i] /= sum;
    }
}

void embed(const Tensor* tokens, const Tensor* embed_weight, Tensor* out) {
    float* o = (float*)out->data;
    int n = tokens->shape[0];
    int embed_size = embed_weight->shape[1];
    int* ids = (int*)tokens->data;

    out->dim = 2;
    out->data_type = TENSOR_DATA_TYPE_F32;
    out->shape[0] = n;
    out->shape[1] = embed_size;

    if(embed_weight->data_type == TENSOR_DATA_TYPE_BF16) {
        uint16_t* embed = (uint16_t*)embed_weight->data;
        #pragma omp parallel for if(n >= 64)
        for(int r=0;r<n;++r) {
            float* ro = o + r*embed_size;
            uint16_t* eo = embed + ids[r]*embed_size;
            for(int i=0;i<embed_size;++i) {
                ro[i] = bf16_to_f32(eo[i]);
            }
        }
    }
    //default is F32
    else {
        float* embed = (float*)embed_weight->data;
        #pragma omp parallel for if(n >= 64)
        for(int i=0;i<n;++i) {
            memcpy(o+i*embed_size,embed + ids[i]*embed_size, sizeof(float)*embed_size);
        }
    }
}

inline void add_inplace(const Tensor* in,const Tensor* param) {
    // assert(in->nelements() == param->nelements());
    float* in_data = (float*)in->data;
    float* param_data = (float*)param->data;
    for(int i=0; i<in->nelements(); ++i) {
        in_data[i] += param_data[i];
    }
}

static inline void simd_gemv_f32(const float* x, const float* weight, float* out, int hidden_dim, int out_dim) {
    const int block_count = out_dim / 4;

    #pragma omp parallel for if(out_dim >= 8192) schedule(static)
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

static inline void simd_gemv2_f32(const float* x,
                                  const float* weight0,
                                  float* out0,
                                  const float* weight1,
                                  float* out1,
                                  int hidden_dim,
                                  int out_dim) {
    const int block_count = out_dim / 4;

    #pragma omp parallel for if(out_dim >= 8192) schedule(static)
    for(int bi = 0; bi < block_count; ++bi) {
        const int oi = bi * 4;
        const float* w00 = weight0 + (oi + 0) * hidden_dim;
        const float* w01 = weight0 + (oi + 1) * hidden_dim;
        const float* w02 = weight0 + (oi + 2) * hidden_dim;
        const float* w03 = weight0 + (oi + 3) * hidden_dim;
        const float* w10 = weight1 + (oi + 0) * hidden_dim;
        const float* w11 = weight1 + (oi + 1) * hidden_dim;
        const float* w12 = weight1 + (oi + 2) * hidden_dim;
        const float* w13 = weight1 + (oi + 3) * hidden_dim;

        __m256 acc00 = _mm256_setzero_ps();
        __m256 acc01 = _mm256_setzero_ps();
        __m256 acc02 = _mm256_setzero_ps();
        __m256 acc03 = _mm256_setzero_ps();
        __m256 acc10 = _mm256_setzero_ps();
        __m256 acc11 = _mm256_setzero_ps();
        __m256 acc12 = _mm256_setzero_ps();
        __m256 acc13 = _mm256_setzero_ps();

        int c = 0;
        for(; c + 7 < hidden_dim; c += 8) {
            __m256 xv = _mm256_loadu_ps(x + c);
            acc00 = _mm256_fmadd_ps(xv, _mm256_loadu_ps(w00 + c), acc00);
            acc01 = _mm256_fmadd_ps(xv, _mm256_loadu_ps(w01 + c), acc01);
            acc02 = _mm256_fmadd_ps(xv, _mm256_loadu_ps(w02 + c), acc02);
            acc03 = _mm256_fmadd_ps(xv, _mm256_loadu_ps(w03 + c), acc03);
            acc10 = _mm256_fmadd_ps(xv, _mm256_loadu_ps(w10 + c), acc10);
            acc11 = _mm256_fmadd_ps(xv, _mm256_loadu_ps(w11 + c), acc11);
            acc12 = _mm256_fmadd_ps(xv, _mm256_loadu_ps(w12 + c), acc12);
            acc13 = _mm256_fmadd_ps(xv, _mm256_loadu_ps(w13 + c), acc13);
        }

        float s00 = hsum256_ps(acc00);
        float s01 = hsum256_ps(acc01);
        float s02 = hsum256_ps(acc02);
        float s03 = hsum256_ps(acc03);
        float s10 = hsum256_ps(acc10);
        float s11 = hsum256_ps(acc11);
        float s12 = hsum256_ps(acc12);
        float s13 = hsum256_ps(acc13);
        for(; c < hidden_dim; ++c) {
            float xv = x[c];
            s00 += xv * w00[c];
            s01 += xv * w01[c];
            s02 += xv * w02[c];
            s03 += xv * w03[c];
            s10 += xv * w10[c];
            s11 += xv * w11[c];
            s12 += xv * w12[c];
            s13 += xv * w13[c];
        }

        out0[oi + 0] = s00;
        out0[oi + 1] = s01;
        out0[oi + 2] = s02;
        out0[oi + 3] = s03;
        out1[oi + 0] = s10;
        out1[oi + 1] = s11;
        out1[oi + 2] = s12;
        out1[oi + 3] = s13;
    }

    for(int oi = block_count * 4; oi < out_dim; ++oi) {
        const float* w0 = weight0 + oi * hidden_dim;
        const float* w1 = weight1 + oi * hidden_dim;
        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        int c = 0;
        for(; c + 7 < hidden_dim; c += 8) {
            __m256 xv = _mm256_loadu_ps(x + c);
            acc0 = _mm256_fmadd_ps(xv, _mm256_loadu_ps(w0 + c), acc0);
            acc1 = _mm256_fmadd_ps(xv, _mm256_loadu_ps(w1 + c), acc1);
        }

        float s0 = hsum256_ps(acc0);
        float s1 = hsum256_ps(acc1);
        for(; c < hidden_dim; ++c) {
            float xv = x[c];
            s0 += xv * w0[c];
            s1 += xv * w1[c];
        }
        out0[oi] = s0;
        out1[oi] = s1;
    }
}

static inline void simd_gemv2_packed4_f32(const float* x,
                                          const float* packed_weight0,
                                          float* out0,
                                          const float* packed_weight1,
                                          float* out1,
                                          int hidden_dim,
                                          int out_dim) {
    const int block_count = out_dim / 4;

    #pragma omp parallel for if(out_dim >= 8192) schedule(static)
    for(int bi = 0; bi < block_count; ++bi) {
        const int oi = bi * 4;
        const float* pw0 = packed_weight0 + oi * hidden_dim;
        const float* pw1 = packed_weight1 + oi * hidden_dim;
        __m128 acc0 = _mm_setzero_ps();
        __m128 acc1 = _mm_setzero_ps();

        int c = 0;
        for(; c + 3 < hidden_dim; c += 4) {
            acc0 = _mm_fmadd_ps(_mm_set1_ps(x[c + 0]), _mm_loadu_ps(pw0 + (c + 0) * 4), acc0);
            acc0 = _mm_fmadd_ps(_mm_set1_ps(x[c + 1]), _mm_loadu_ps(pw0 + (c + 1) * 4), acc0);
            acc0 = _mm_fmadd_ps(_mm_set1_ps(x[c + 2]), _mm_loadu_ps(pw0 + (c + 2) * 4), acc0);
            acc0 = _mm_fmadd_ps(_mm_set1_ps(x[c + 3]), _mm_loadu_ps(pw0 + (c + 3) * 4), acc0);

            acc1 = _mm_fmadd_ps(_mm_set1_ps(x[c + 0]), _mm_loadu_ps(pw1 + (c + 0) * 4), acc1);
            acc1 = _mm_fmadd_ps(_mm_set1_ps(x[c + 1]), _mm_loadu_ps(pw1 + (c + 1) * 4), acc1);
            acc1 = _mm_fmadd_ps(_mm_set1_ps(x[c + 2]), _mm_loadu_ps(pw1 + (c + 2) * 4), acc1);
            acc1 = _mm_fmadd_ps(_mm_set1_ps(x[c + 3]), _mm_loadu_ps(pw1 + (c + 3) * 4), acc1);
        }

        float tmp0[4];
        float tmp1[4];
        _mm_storeu_ps(tmp0, acc0);
        _mm_storeu_ps(tmp1, acc1);
        for(; c < hidden_dim; ++c) {
            __m128 w0 = _mm_loadu_ps(pw0 + c * 4);
            __m128 w1 = _mm_loadu_ps(pw1 + c * 4);
            acc0 = _mm_fmadd_ps(_mm_set1_ps(x[c]), w0, acc0);
            acc1 = _mm_fmadd_ps(_mm_set1_ps(x[c]), w1, acc1);
        }
        _mm_storeu_ps(tmp0, acc0);
        _mm_storeu_ps(tmp1, acc1);

        out0[oi + 0] = tmp0[0];
        out0[oi + 1] = tmp0[1];
        out0[oi + 2] = tmp0[2];
        out0[oi + 3] = tmp0[3];
        out1[oi + 0] = tmp1[0];
        out1[oi + 1] = tmp1[1];
        out1[oi + 2] = tmp1[2];
        out1[oi + 3] = tmp1[3];
    }

}

void matmul(Tensor* tb,const Tensor* weight , Tensor* out);

static inline void matmul_pair(Tensor* tb,
                               const Tensor* weight0,
                               Tensor* out0,
                               const Tensor* weight1,
                               Tensor* out1) {
    assert(tb->dim == weight0->dim);
    assert(tb->dim == weight1->dim);

    int hidden_size = tb->shape[tb->dim - 1];
    assert(hidden_size == weight0->shape[weight0->dim - 1]);
    assert(hidden_size == weight1->shape[weight1->dim - 1]);
    assert(weight0->shape[weight0->dim - 2] == weight1->shape[weight1->dim - 2]);

    float* data = (float*)tb->data;
    float* weight0_data = (float*)weight0->data;
    float* weight1_data = (float*)weight1->data;
    float* out0_data = (float*)out0->data;
    float* out1_data = (float*)out1->data;

    out0->dim = tb->dim;
    out1->dim = tb->dim;
    memcpy(out0->shape, tb->shape, sizeof(out0->shape));
    memcpy(out1->shape, tb->shape, sizeof(out1->shape));
    out0->shape[out0->dim - 1] = weight0->shape[weight0->dim - 2];
    out1->shape[out1->dim - 1] = weight1->shape[weight1->dim - 2];
    out0->data_type = tb->data_type;
    out1->data_type = tb->data_type;

    int row = tb->nelements() / hidden_size;
    int out_size = out0->shape[out0->dim - 1];

    if(row == 1) {
        #if defined(__AVX2__) && defined(__FMA__)
        const bool use_packed =
            weight0->packed_data != nullptr && weight0->packed_block == 4 &&
            weight1->packed_data != nullptr && weight1->packed_block == 4;
        if (use_packed) {
            simd_gemv2_packed4_f32(data,
                                   (const float*)weight0->packed_data,
                                   out0_data,
                                   (const float*)weight1->packed_data,
                                   out1_data,
                                   hidden_size,
                                   out_size);
            for(int oi = (out_size / 4) * 4; oi < out_size; ++oi) {
                float s0 = 0.0f;
                float s1 = 0.0f;
                float* w0 = weight0_data + oi * hidden_size;
                float* w1 = weight1_data + oi * hidden_size;
                for(int i = 0; i < hidden_size; ++i) {
                    s0 += data[i] * w0[i];
                    s1 += data[i] * w1[i];
                }
                out0_data[oi] = s0;
                out1_data[oi] = s1;
            }
        } else {
            simd_gemv2_f32(data, weight0_data, out0_data, weight1_data, out1_data, hidden_size, out_size);
        }
        #else
        for(int k = 0; k < out_size; ++k) {
            float s0 = 0.0f;
            float s1 = 0.0f;
            float* w0 = weight0_data + k * hidden_size;
            float* w1 = weight1_data + k * hidden_size;
            for(int i = 0; i < hidden_size; ++i) {
                s0 += data[i] * w0[i];
                s1 += data[i] * w1[i];
            }
            out0_data[k] = s0;
            out1_data[k] = s1;
        }
        #endif
        return;
    }

    matmul(tb, weight0, out0);
    matmul(tb, weight1, out1);
}


/*
  Tensor matrix multiply like x@Q
  tensor x shape: [S,H]
  tensor Q shape: [out_dim,H]
*/
void matmul(Tensor* tb,const Tensor* weight , Tensor* out) {
    //tensor matmul at the last two dimension
    assert(tb->dim == weight->dim);
    
    int hidden_size = tb->shape[tb->dim-1];
    assert(hidden_size == weight->shape[weight->dim-1]);

    float* data = (float*)tb->data;
    float* weight_data = (float*)weight->data;
    float* out_data = (float*)out->data;
    
    out->dim = tb->dim;
    memcpy(out->shape, tb->shape, sizeof(out->shape));
    out->shape[out->dim-1] = weight->shape[weight->dim-2];
    out->data_type = tb->data_type;
    
    int row = tb->nelements() / hidden_size;
    int out_size = out->shape[out->dim-1];
    if(row == 1) {
        #if defined(__AVX2__) && defined(__FMA__)
        simd_gemv_f32(data, weight_data, out_data, hidden_size, out_size);
        #else
        for(int k=0; k<out_size; ++k) {
            float* w = weight_data + k * hidden_size;
            float s = 0.0f;
            for(int i=0; i<hidden_size; ++i) {
                s += data[i] * w[i];
            }
            out_data[k] = s;
        }
        #endif
        return;
    }

    #pragma omp parallel for if(row >= 8)
    for(int r=0; r<row; ++r) {
        float* o = out_data + r*out_size;
        float* x = data + r*hidden_size;

        #if defined(__AVX2__) && defined(__FMA__)
        int k = 0;
        for(; k + 3 < out_size; k += 4) {
            float* w0 = weight_data + (k + 0) * hidden_size;
            float* w1 = weight_data + (k + 1) * hidden_size;
            float* w2 = weight_data + (k + 2) * hidden_size;
            float* w3 = weight_data + (k + 3) * hidden_size;

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
            float* w = weight_data + k * hidden_size;
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
            float* w = weight_data + k*hidden_size;
            float ss = 0.0f;
            for(int i=0; i<hidden_size; ++i) {
                ss += x[i] * w[i];
            }
            o[k] = ss;
        }
        #endif
    }
}


void rms_norm(const Tensor* tb, const Tensor* norm_weight, Tensor* out, float eps) {
    int d = norm_weight->shape[0];
    int rows = tb->nelements() / d;

    float* weight = (float*)norm_weight->data;
    float* data = (float*)tb->data;
    float* out_data = nullptr;

    // if out is nullptr, do inplace norm, otherwise write to out
    if(out == nullptr) {
        out_data = (float*)tb->data;
    }
    else {
        out_data = (float*)out->data;
        out->data_type = tb->data_type;
        out->dim = tb->dim;
        memcpy(out->shape,tb->shape,sizeof(out->shape));
    }

    for(size_t i=0; i<rows; ++i) {
        float* x = data + i*d;
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
void rope(Tensor* tb,float theta_rope, int seq_len_processed) {
    int row = tb->shape[0];
    int out_dim = tb->nelements() / row;

    float* data = (float*)tb->data;
    //int32_t* pos_data = (int32_t*)pos->data;

    #pragma omp parallel for if (row >= 8)
    for(int r=0;r<row;++r) {
        for(int i=0;i<out_dim;i+=2)  {
            float theta = (seq_len_processed+r) / powf(theta_rope,2*i / out_dim);
            float x0 = data[i];
            float x1 = data[i+1]; 
            data[i] = x0 * cosf(theta) - x1 * sinf(theta);
            data[i+1] = x0 * sinf(theta) + x1 * cosf(theta);
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
    float* kdata = (float*)k->data;
    float* vdata = (float*)v->data;
    uint16_t* kdata_bf16 = (uint16_t*)k->data;
    uint16_t* vdata_bf16 = (uint16_t*)v->data;
    const bool k_is_bf16 = k->data_type == TENSOR_DATA_TYPE_BF16;
    const bool v_is_bf16 = v->data_type == TENSOR_DATA_TYPE_BF16;

    if (q_len == 1) {
        const int total_kv = seq_len_processed + 1;
        #pragma omp parallel for if(q_head >= 8)
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
                float* k_data0 = kdata + base0;
                float* k_data1 = kdata + base1;
                float* k_data2 = kdata + base2;
                float* k_data3 = kdata + base3;
                uint16_t* k_data0_bf16 = kdata_bf16 + base0;
                uint16_t* k_data1_bf16 = kdata_bf16 + base1;
                uint16_t* k_data2_bf16 = kdata_bf16 + base2;
                uint16_t* k_data3_bf16 = kdata_bf16 + base3;

                #if defined(__AVX2__) && defined(__FMA__)
                __m256 acc0 = _mm256_setzero_ps();
                __m256 acc1 = _mm256_setzero_ps();
                __m256 acc2 = _mm256_setzero_ps();
                __m256 acc3 = _mm256_setzero_ps();
                int i = 0;
                for(; i + 7 < head_dim; i += 8) {
                    __m256 qv = _mm256_loadu_ps(q_data + i);
                    __m256 k0 = k_is_bf16 ? load_bf16_8_as_f32(k_data0_bf16 + i) : _mm256_loadu_ps(k_data0 + i);
                    __m256 k1 = k_is_bf16 ? load_bf16_8_as_f32(k_data1_bf16 + i) : _mm256_loadu_ps(k_data1 + i);
                    __m256 k2 = k_is_bf16 ? load_bf16_8_as_f32(k_data2_bf16 + i) : _mm256_loadu_ps(k_data2 + i);
                    __m256 k3 = k_is_bf16 ? load_bf16_8_as_f32(k_data3_bf16 + i) : _mm256_loadu_ps(k_data3 + i);
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
                    s0 += qv * (k_is_bf16 ? bf16_to_f32(k_data0_bf16[i]) : k_data0[i]);
                    s1 += qv * (k_is_bf16 ? bf16_to_f32(k_data1_bf16[i]) : k_data1[i]);
                    s2 += qv * (k_is_bf16 ? bf16_to_f32(k_data2_bf16[i]) : k_data2[i]);
                    s3 += qv * (k_is_bf16 ? bf16_to_f32(k_data3_bf16[i]) : k_data3[i]);
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
                float* k_data = kdata + base;
                uint16_t* k_data_bf16 = kdata_bf16 + base;
                #if defined(__AVX2__) && defined(__FMA__)
                __m256 acc = _mm256_setzero_ps();
                int i = 0;
                for(; i + 7 < head_dim; i += 8) {
                    __m256 qv = _mm256_loadu_ps(q_data + i);
                    __m256 kv = k_is_bf16 ? load_bf16_8_as_f32(k_data_bf16 + i) : _mm256_loadu_ps(k_data + i);
                    acc = _mm256_fmadd_ps(qv, kv, acc);
                }
                float s = hsum256_ps(acc);
                for(; i < head_dim; ++i) {
                    s += q_data[i] * (k_is_bf16 ? bf16_to_f32(k_data_bf16[i]) : k_data[i]);
                }
                #else
                float s = 0.0f;
                for(int i = 0; i < head_dim; ++i) {
                    s += q_data[i] * (k_is_bf16 ? bf16_to_f32(k_data_bf16[i]) : k_data[i]);
                }
                #endif
                attn_data[kv_index] = s * scale;
            }

            softmax(attn_data, total_kv);

            memset(q_data, 0, sizeof(float) * head_dim);
            for(int kv_index = 0; kv_index < total_kv; ++kv_index) {
                int base = kv_index * head_dim * kv_head + kv_selected * head_dim;
                float* v_data = vdata + base;
                uint16_t* v_data_bf16 = vdata_bf16 + base;
                #if defined(__AVX2__) && defined(__FMA__)
                __m256 attn_vec = _mm256_set1_ps(attn_data[kv_index]);
                int i = 0;
                for(; i + 7 < head_dim; i += 8) {
                    __m256 outv = _mm256_loadu_ps(q_data + i);
                    __m256 vv = v_is_bf16 ? load_bf16_8_as_f32(v_data_bf16 + i) : _mm256_loadu_ps(v_data + i);
                    outv = _mm256_fmadd_ps(attn_vec, vv, outv);
                    _mm256_storeu_ps(q_data + i, outv);
                }
                for(; i < head_dim; ++i) {
                    q_data[i] += attn_data[kv_index] * (v_is_bf16 ? bf16_to_f32(v_data_bf16[i]) : v_data[i]);
                }
                #else
                for(int i = 0; i < head_dim; ++i) {
                    q_data[i] += attn_data[kv_index] * (v_is_bf16 ? bf16_to_f32(v_data_bf16[i]) : v_data[i]);
                }
                #endif
            }
        }

        q->reshape_2d();
        return;
    }

    #pragma omp parallel for if(q_head >= 8)
    for(int h=0;h<q_head;++h) {
        int kv_selected = h / group_size;
        for(int q_index=0;q_index<q_len;++q_index) {
            float* q_data = qdata + q_index * head_dim * q_head + h * head_dim;

            float* attn_data = attn + h * (q_len + seq_len_processed); 
            
            int kv_index = 0;
            
            // q can only attention the kv before current q_index
            for(; kv_index + 3 <=(q_index + seq_len_processed); kv_index+=4) {
                int base0 = (kv_index + 0) * head_dim * kv_head + kv_selected * head_dim;
                int base1 = (kv_index + 1) * head_dim * kv_head + kv_selected * head_dim;
                int base2 = (kv_index + 2) * head_dim * kv_head + kv_selected * head_dim;
                int base3 = (kv_index + 3) * head_dim * kv_head + kv_selected * head_dim;
                float* k_data0 = kdata + base0;
                float* k_data1 = kdata + base1;
                float* k_data2 = kdata + base2;
                float* k_data3 = kdata + base3;
                uint16_t* k_data0_bf16 = kdata_bf16 + base0;
                uint16_t* k_data1_bf16 = kdata_bf16 + base1;
                uint16_t* k_data2_bf16 = kdata_bf16 + base2;
                uint16_t* k_data3_bf16 = kdata_bf16 + base3;

                __m256 acc0 = _mm256_setzero_ps();
                __m256 acc1 = _mm256_setzero_ps();
                __m256 acc2 = _mm256_setzero_ps();
                __m256 acc3 = _mm256_setzero_ps();

                int i = 0;
                for(; i + 7 < head_dim; i += 8) {
                    __m256 qv = _mm256_loadu_ps(q_data + i);
                    __m256 k0 = k_is_bf16 ? load_bf16_8_as_f32(k_data0_bf16 + i) : _mm256_loadu_ps(k_data0 + i);
                    __m256 k1 = k_is_bf16 ? load_bf16_8_as_f32(k_data1_bf16 + i) : _mm256_loadu_ps(k_data1 + i);
                    __m256 k2 = k_is_bf16 ? load_bf16_8_as_f32(k_data2_bf16 + i) : _mm256_loadu_ps(k_data2 + i);
                    __m256 k3 = k_is_bf16 ? load_bf16_8_as_f32(k_data3_bf16 + i) : _mm256_loadu_ps(k_data3 + i);
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
                    s0 += qv * (k_is_bf16 ? bf16_to_f32(k_data0_bf16[i]) : k_data0[i]);
                    s1 += qv * (k_is_bf16 ? bf16_to_f32(k_data1_bf16[i]) : k_data1[i]);
                    s2 += qv * (k_is_bf16 ? bf16_to_f32(k_data2_bf16[i]) : k_data2[i]);
                    s3 += qv * (k_is_bf16 ? bf16_to_f32(k_data3_bf16[i]) : k_data3[i]);
                }
                attn_data[kv_index+0] = s0 * scale;
                attn_data[kv_index+1] = s1 * scale;
                attn_data[kv_index+2] = s2 * scale;
                attn_data[kv_index+3] = s3 * scale;
            }

            for(; kv_index <= (q_index + seq_len_processed); ++kv_index) {
                int base = kv_index * head_dim * kv_head + kv_selected * head_dim;
                float* k_data = kdata + base;
                uint16_t* k_data_bf16 = kdata_bf16 + base;

                __m256 acc = _mm256_setzero_ps();
                int i = 0;  
                for(; i + 7 < head_dim; i += 8) {
                    __m256 qv = _mm256_loadu_ps(q_data + i);
                    __m256 kv = k_is_bf16 ? load_bf16_8_as_f32(k_data_bf16 + i) : _mm256_loadu_ps(k_data + i);
                    acc = _mm256_fmadd_ps(qv, kv, acc);
                }
                float s = hsum256_ps(acc);
                for(; i < head_dim; ++i) {
                    s += q_data[i] * (k_is_bf16 ? bf16_to_f32(k_data_bf16[i]) : k_data[i]);
                }
                attn_data[kv_index] = s * scale;
            }

            /**
            for(int kv_index=0; kv_index<=(q_index + seq_len_processed); ++kv_index) {
                float* k_data = kdata + kv_index * head_dim * kv_head + kv_selected * head_dim;
                float sum = 0.0f;
                for(int i=0;i<head_dim;++i) {
                    sum += q_data[i] * k_data[i];
                }
                attn_data[kv_index] = sum * scale;
            }
            */
            
            softmax(attn_data, q_index+1+seq_len_processed);
        
            memset(q_data,0,sizeof(float) * head_dim);
            for(int kv_index=0; kv_index<=(q_index + seq_len_processed); ++kv_index) {
                int base = kv_index * head_dim * kv_head + kv_selected * head_dim;
                float* v_data = vdata + base;
                uint16_t* v_data_bf16 = vdata_bf16 + base;
                #if defined(__AVX2__) && defined(__FMA__)
                __m256 attn_vec = _mm256_set1_ps(attn_data[kv_index]);
                int i = 0;
                for(; i + 7 < head_dim; i += 8) {
                    __m256 outv = _mm256_loadu_ps(q_data + i);
                    __m256 vv = v_is_bf16 ? load_bf16_8_as_f32(v_data_bf16 + i) : _mm256_loadu_ps(v_data + i);
                    outv = _mm256_fmadd_ps(attn_vec, vv, outv);
                    _mm256_storeu_ps(q_data + i, outv);
                }
                for(; i < head_dim; ++i) {
                    q_data[i] += attn_data[kv_index] * (v_is_bf16 ? bf16_to_f32(v_data_bf16[i]) : v_data[i]);
                }
                #else
                for(int i=0;i<head_dim;++i) {
                    q_data[i] += attn_data[kv_index] * (v_is_bf16 ? bf16_to_f32(v_data_bf16[i]) : v_data[i]);
                }
                #endif
            }
        }
    }
    
    // now q is softmaxt(q@k.T/sqrt(d))@V shape [sequence,q_heads,head_size]
    q->reshape_2d();
}


void mlp(Tensor* up, Tensor* gate) {
    float* g_data = (float*)gate->data;
    float* u_data = (float*)up->data; 
    for(int i=0;i<up->nelements(); ++i) {
        float silu = g_data[i] * (1.0f / (1 + expf(-g_data[i])));
        u_data[i] *= silu;
    }
}
