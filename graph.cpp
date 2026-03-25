
#include "graph.h"
#include "runtime.h"


Tensor* OperatorGraph::build_embed(const Tensor* in, const Tensor* embed, Tensor* out) {
    TOperator op = TOperator();
    op.operator_type = OP_TENSOR_EMBED;
    op.params[0] = in;
    op.params[1] = embed;
    op.out = out;
    graph.push_back(op);
    return op.out;
}


Tensor* OperatorGraph::build_norm(const Tensor* in,const Tensor* norm_weight, Tensor* out) {
    TOperator op = TOperator();
    op.operator_type = OP_TENSOR_NORM;
    op.params[0] = in;
    op.params[1] = norm_weight;
    op.out = out;
    graph.push_back(op);
    return op.out;
}

Tensor* OperatorGraph::build_reshape_3d(Tensor* inout, int last_dim) {
    TOperator op = TOperator();
    op.operator_type = OP_TENSOR_RESHAPE_3D;
    op.params[0] = inout;
    op.other_params[0] = last_dim;
    op.out = inout;
    graph.push_back(op);
    return op.out;
}


Tensor* OperatorGraph::build_matmul(const Tensor* in1,const Tensor* in2, Tensor* out) {
    TOperator op = TOperator();
    op.operator_type = OP_TENSOR_MATMUL;
    op.params[0] = in1;
    op.params[1] = in2;
    op.out = out;
    //Tensor* out = mem->allocate_out();
    //memcpy(out,in1,sizeof(Tensor));
    //out->shape[out->dim-1] = in2->shape[in2->dim-2];
    //out->data = ???;
    graph.push_back(op);
    return op.out;
}

Tensor* OperatorGraph::build_rope(Tensor* in, Tensor* pos) {
    TOperator op = TOperator();
    op.operator_type = OP_TENSOR_ROPE;
    op.params[0] = in;
    op.params[1] = pos;
    // rope is inplace
    op.out = in;
    return op.out;
}


Tensor* OperatorGraph::build_attention(const Tensor* q, const Tensor* k,const Tensor* v,
                                        Tensor* att_score, Tensor * out) {
    TOperator op = TOperator();
    op.operator_type = OP_TENSOR_ATTENTION;
    op.params[0] = q,
    op.params[1] = k;
    op.params[2] = v;
    op.params[3] = att_score;

    op.out = out;
    return op.out;
}


Tensor* OperatorGraph::build_add(const Tensor*in,const Tensor* param, Tensor* out) {
    TOperator op = TOperator();
    op.operator_type = OP_TENSOR_ADD;
    op.params[0] = in;
    op.params[1] = param;
    op.out = out;
    graph.push_back(op);
    return op.out;
}

Tensor* OperatorGraph::build_mlp(const Tensor* in, Tensor* up_out,Tensor* gate_out, Tensor* ffn_out) {
    TOperator op = TOperator();
    op.operator_type = OP_TENSOR_MLP;
    op.params[0] = in;
    op.params[1] = up_out;
    op.params[2] = gate_out;
    op.out = ffn_out;
    graph.push_back(op);
    return op.out;
}


void OperatorGraph::forward() {
    for(TOperator& op : graph) {
        op.forward();
    }
}