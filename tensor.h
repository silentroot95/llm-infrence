#pragma once

#include <string.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

const int MAX_TENSOR_DIM = 4;

enum TENSOR_DATA_TYPE {
    TENSOR_DATA_TYPE_INT32,
    TENSOR_DATA_TYPE_F32,
    TENSOR_DATA_TYPE_F16,
    TENSOR_DATA_TYPE_BF16,
    TENSOR_DATA_TYPE_UNKNOWN
};


inline TENSOR_DATA_TYPE get_data_type(const char* data_typr_str) {
    if (0 == strcmp(data_typr_str,"F32"))   {
        return TENSOR_DATA_TYPE_F32;
    }
    if (0 == strcmp(data_typr_str,"F16")) {
        return TENSOR_DATA_TYPE_F16;
    }
    if (0 == strcmp(data_typr_str,"BF16"))
    {
        return TENSOR_DATA_TYPE_BF16;
    }
    return TENSOR_DATA_TYPE_UNKNOWN;
}

inline size_t get_data_type_size(TENSOR_DATA_TYPE data_type) {
    switch(data_type) {
        case TENSOR_DATA_TYPE_F32:
        case TENSOR_DATA_TYPE_INT32:
            return 4;
        case TENSOR_DATA_TYPE_F16:
        case TENSOR_DATA_TYPE_BF16:
            return 2;
        default:
            return 0;
    }
}

inline float bf16_to_f32(uint16_t bf16) {
    union {
        uint32_t u32;
        float f32;
    } u;
    
    u.u32 = ((uint32_t)bf16) << 16;
    return u.f32;
}

inline uint16_t f32_to_bf16(float f32) {
    union {
        uint32_t u32;
        float f32;
    } u;

    u.f32 = f32;
    return (uint16_t)(u.u32 >> 16);
}

struct Tensor {
    TENSOR_DATA_TYPE data_type;
    int dim;
    int shape[MAX_TENSOR_DIM];
    void* data;
    void* packed_data;
    int packed_block;

    inline int nelements() const {
        if(dim<=0) return 0;
        int num = 1;
        for(int i=0;i<dim;++i){
            num *= shape[i];
        }
        return num;
    }

    inline size_t nbytes() const {
        size_t dsize = get_data_type_size(data_type);
        int num = nelements();
        return num*dsize;
    }

    inline Tensor peek_at(int index) {
        Tensor res;
        memcpy(&res, this, sizeof(res));

        float* f = (float*)data;
        int row_stride = nelements() / shape[0];

        res.shape[0] = 1;
        res.data = (void*)(f + index * row_stride);
        return res;
    }

    inline void reshape_2d() {
        if(dim < 2) return;
        shape[1] = nelements() / shape[0];
        dim = 2;
    }

    inline void reshape_3d(int last_dim_size) {
        if(dim < 1) return;
        int num = nelements();
        int s1 = num / shape[0] / last_dim_size;
        if(s1*last_dim_size*shape[0] != num) {
            fprintf(stderr,"Error reshape_3d, not divisible!\n");
            exit(EXIT_FAILURE);
        };

        shape[2] = last_dim_size;
        shape[1] =  s1;
        dim = 3;
    }
};


