
#include <vector>
#include "operator.h"


struct Tensor;


struct OperatorGraph {
    Tensor* build_embed(const Tensor* in,const Tensor* embed, Tensor* out);
    Tensor* build_add(const Tensor*in,const Tensor* param, Tensor* out);
    Tensor* build_norm(const Tensor* in,const Tensor* norm_weight, Tensor* out);
    Tensor* build_reshape_3d(Tensor* inout, int last_dim);
    Tensor* build_matmul(const Tensor* in, const Tensor* in2, Tensor* out);
    Tensor* build_rope(Tensor* in, Tensor* pos);
    Tensor* build_attention(const Tensor* q, const Tensor* k,const Tensor* v,
                            Tensor* att_score, Tensor * out);
    Tensor* build_mlp(const Tensor* in, Tensor* up_out,Tensor* gate_out, Tensor* ffn_out);

    void forward();

    std::vector<TOperator> graph;
};
