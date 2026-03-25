
#include <assert.h>

#include "model.h"
#include "mmap.h"

#include "cJSON.h"

static void init_tensor_layout(Tensor* t) {
    t->packed_data = nullptr;
    t->packed_block = 0;
}

static void pack_weight_4col(Tensor* weight) {
    if (weight->data_type != TENSOR_DATA_TYPE_F32 || weight->dim != 2) {
        return;
    }

    const int out_dim = weight->shape[0];
    const int hidden_size = weight->shape[1];
    const int block = 4;
    const int padded_out = ((out_dim + block - 1) / block) * block;

    float* src = (float*)weight->data;
    float* packed = (float*)malloc((size_t)padded_out * hidden_size * sizeof(float));
    if (!packed) {
        fprintf(stderr, "malloc packed weight failed!\n");
        exit(EXIT_FAILURE);
    }

    for (int ob = 0; ob < padded_out; ob += block) {
        float* dst = packed + ob * hidden_size;
        for (int i = 0; i < hidden_size; ++i) {
            for (int lane = 0; lane < block; ++lane) {
                const int out_idx = ob + lane;
                dst[i * block + lane] = out_idx < out_dim ? src[out_idx * hidden_size + i] : 0.0f;
            }
        }
    }

    weight->packed_data = packed;
    weight->packed_block = block;
}


void ModelConfig::load_config(const char* path) {
    size_t size = 0;
    void* data = memory_map(path, &size);

    char* json_string = (char*)malloc(size + 1);
    memcpy(json_string,data,size);
    json_string[size] = '\0';

    memory_unmap(data, size);

    cJSON* json = cJSON_Parse(json_string);
    if (json==NULL){
        fprintf(stderr,"Error parsing JSON: %s\n", cJSON_GetErrorPtr());
        free(json_string);
        exit(EXIT_FAILURE);
    }
    cJSON* item = cJSON_GetObjectItemCaseSensitive(json,"head_dim");
    head_size = item->valueint;
    item = cJSON_GetObjectItemCaseSensitive(json,"hidden_size");
    embed_size = item->valueint;
    item = cJSON_GetObjectItemCaseSensitive(json,"intermediate_size");
    ffn_size = item->valueint;
    item = cJSON_GetObjectItemCaseSensitive(json,"num_hidden_layers");
    num_layers = item->valueint;
    item = cJSON_GetObjectItemCaseSensitive(json,"num_attention_heads");
    q_heads = item->valueint;
    item = cJSON_GetObjectItemCaseSensitive(json,"num_key_value_heads");        
    kv_heads = item->valueint;
    item = cJSON_GetObjectItemCaseSensitive(json,"vocab_size");
    vocab_size = item->valueint;
    item = cJSON_GetObjectItemCaseSensitive(json,"rms_norm_eps"); 
    rms_norm_eps = item->valuedouble;
    item = cJSON_GetObjectItemCaseSensitive(json,"bos_token_id"); 
    bos_token_id = item->valueint;
    item = cJSON_GetObjectItemCaseSensitive(json,"eos_token_id"); 
    eos_token_id = item->valueint;

    item = cJSON_GetObjectItemCaseSensitive(json,"rope_theta"); 
    rope_theta = item->valuedouble;
    
    // we use this for max_seq_len,
    item = cJSON_GetObjectItemCaseSensitive(json,"max_position_embeddings"); 
    max_seq_len = item->valueint;

    free(json_string);
    cJSON_Delete(json);
}


static void copy_to_f32(const cJSON* json,const char* name,void* begin_ptr, void* dst) {
    cJSON* tensor_info = cJSON_GetObjectItemCaseSensitive(json,name);
    cJSON* dtype = cJSON_GetObjectItemCaseSensitive(tensor_info,"dtype");
    cJSON* data_offsets = cJSON_GetObjectItemCaseSensitive(tensor_info, "data_offsets");

    assert(cJSON_IsArray(data_offsets));
    assert(cJSON_GetArraySize(data_offsets)==2);

    uint64_t offset = cJSON_GetArrayItem(data_offsets,0)->valuedouble;
    uint64_t end_offset = cJSON_GetArrayItem(data_offsets,1)->valuedouble;
    uint8_t* tensor_data_ptr = (uint8_t*)begin_ptr  + offset;

    uint64_t nbytes = end_offset - offset;
    TENSOR_DATA_TYPE data_type = get_data_type(dtype->valuestring);
    uint64_t num = nbytes / get_data_type_size(data_type);

    float* dst_data = (float*)dst;

    switch (data_type){
    case TENSOR_DATA_TYPE_F32:
        memcpy(dst,tensor_data_ptr,nbytes);
        break;

    case TENSOR_DATA_TYPE_BF16:{
        uint16_t* src = (uint16_t*)tensor_data_ptr; 
        for(int i=0;i<num;++i) {
            dst_data[i] = bf16_to_f32(src[i]);
        }
        break;
    }
    default:
        fprintf(stderr,"Not Supported!\n");
        exit(EXIT_FAILURE);
        break;
    }
}


void ModelWeights::init(const ModelConfig* config) {
    //model_config = &config;
    TENSOR_DATA_TYPE dtype = TENSOR_DATA_TYPE_F32;
    size_t dtype_size = get_data_type_size(dtype);

    size_t total_elements = (
        2*config->embed_size * config->vocab_size  //embed and lm_head 
        + config->embed_size                      //final_norm
        + config->num_layers*(
            2*config->embed_size                                             //attntion norm and ffn norm
            + 2*config->q_heads * config->head_size * config->embed_size       //q weight and o weight
            + 2*config->kv_heads * config->head_size * config->embed_size      //k v weight
            + 2 * config->head_size                                          //q k norm
            + 3 * config->embed_size * config->ffn_size                       //up gate down weight
        )
    );
    data = malloc(dtype_size * total_elements);

    if(!data) {
        fprintf(stderr,"malloc failed!\n");
        exit(EXIT_FAILURE);
    }
    
    uint8_t* ptr = (uint8_t*)data;

    embeds.data_type = dtype;
    embeds.dim = 2;
    embeds.shape[0] = config->vocab_size;
    embeds.shape[1] = config->embed_size;
    memcpy(&lm_heads,&embeds,sizeof(lm_heads));
    init_tensor_layout(&embeds);
    embeds.data = (void*)ptr;
    ptr += embeds.nbytes();
    
    lm_heads.data = (void*)ptr;
    init_tensor_layout(&lm_heads);
    ptr += lm_heads.nbytes();

    output_norm.data_type = dtype;
    output_norm.dim = 1;
    output_norm.shape[0] = config->embed_size;
    output_norm.data = (void*)ptr;
    init_tensor_layout(&output_norm);
    ptr += output_norm.nbytes();

    
    layers = (TransformerLayer*)malloc(sizeof(TransformerLayer)*config->num_layers);
    num_layers = config->num_layers;
    if(!layers) {
        fprintf(stderr,"malloc TransformLayer failed!\n");
        exit(EXIT_FAILURE);
    }

    for(int i=0;i<config->num_layers;++i) {
        layers[i].attn_norm.data_type = dtype;
        layers[i].attn_norm.dim = 1;
        layers[i].attn_norm.shape[0] = config->embed_size;
        layers[i].attn_norm.data = (void*)ptr;
        init_tensor_layout(&layers[i].attn_norm);
        ptr += layers[i].attn_norm.nbytes();

        layers[i].q_weight.data_type = dtype;
        layers[i].q_weight.dim = 2;
        layers[i].q_weight.shape[0] = config->q_heads * config->head_size;
        layers[i].q_weight.shape[1] = config->embed_size;
        layers[i].q_weight.data = (void*)ptr;
        init_tensor_layout(&layers[i].q_weight);
        ptr += layers[i].q_weight.nbytes();

        layers[i].k_weight.data_type = dtype;
        layers[i].k_weight.dim = 2;
        layers[i].k_weight.shape[0] = config->kv_heads * config->head_size;
        layers[i].k_weight.shape[1] = config->embed_size;
        layers[i].k_weight.data = (void*)ptr;
        init_tensor_layout(&layers[i].k_weight);
        ptr += layers[i].k_weight.nbytes();

        memcpy(&layers[i].v_weight,&layers[i].k_weight,sizeof(Tensor));
        layers[i].v_weight.data = (void*)ptr;
        init_tensor_layout(&layers[i].v_weight);
        ptr += layers[i].v_weight.nbytes();

        layers[i].o_weight.data_type = dtype;
        layers[i].o_weight.dim = 2;
        layers[i].o_weight.shape[0] = config->embed_size;
        layers[i].o_weight.shape[1] = config->q_heads * config->head_size;
        layers[i].o_weight.data = (void*)ptr;
        init_tensor_layout(&layers[i].o_weight);
        ptr += layers[i].o_weight.nbytes();

        layers[i].q_norm.data_type = dtype;
        layers[i].q_norm.dim = 1;
        layers[i].q_norm.shape[0] = config->head_size;
        layers[i].q_norm.data = (void*)ptr;
        init_tensor_layout(&layers[i].q_norm);
        ptr += layers[i].q_norm.nbytes();

        memcpy(&layers[i].k_norm,&layers[i].q_norm,sizeof(Tensor));
        layers[i].k_norm.data = (void*)ptr;
        init_tensor_layout(&layers[i].k_norm);
        ptr += layers[i].k_norm.nbytes();

        memcpy(&layers[i].ffn_norm,&layers[i].attn_norm,sizeof(Tensor));
        layers[i].ffn_norm.data = (void*)ptr;
        init_tensor_layout(&layers[i].ffn_norm);
        ptr += layers[i].ffn_norm.nbytes();
        
        layers[i].up_weight.data_type = dtype;
        layers[i].up_weight.dim = 2;
        layers[i].up_weight.shape[0] = config->ffn_size;
        layers[i].up_weight.shape[1] = config->embed_size;
        layers[i].up_weight.data = (void*)ptr;
        init_tensor_layout(&layers[i].up_weight);
        ptr += layers[i].up_weight.nbytes();

        memcpy(&layers[i].gate_weight,&layers[i].up_weight,sizeof(Tensor));
        layers[i].gate_weight.data = (void*)ptr;
        init_tensor_layout(&layers[i].gate_weight);
        ptr += layers[i].gate_weight.nbytes();

        layers[i].down_weight.data_type = dtype;
        layers[i].down_weight.dim = 2;
        layers[i].down_weight.shape[0] = config->embed_size;
        layers[i].down_weight.shape[1] = config->ffn_size;
        layers[i].down_weight.data = (void*)ptr;
        init_tensor_layout(&layers[i].down_weight);
        ptr += layers[i].down_weight.nbytes();
        
    }
}

inline void ModelWeights::destory() {
    free(lm_heads.packed_data);
    lm_heads.packed_data = nullptr;
    for (int i = 0; i < num_layers; ++i) {
        free(layers[i].q_weight.packed_data);
        free(layers[i].k_weight.packed_data);
        free(layers[i].v_weight.packed_data);
        free(layers[i].o_weight.packed_data);
        free(layers[i].up_weight.packed_data);
        free(layers[i].gate_weight.packed_data);
        free(layers[i].down_weight.packed_data);
    }
    free(layers);
    free(data);
}

void ModelWeights::pack_hot_weights() {
    for (int i = 0; i < num_layers; ++i) {
        pack_weight_4col(&layers[i].up_weight);
        pack_weight_4col(&layers[i].gate_weight);
    }
}


void ModelWeights::load_tensor(const char* path) {
    size_t size = 0;
    void* data = memory_map(path, &size);

    uint64_t header_size = *((uint64_t*)data);

    char* json_string = (char*)malloc(header_size+1);
    memcpy(json_string, (char*)data+8, header_size);
    json_string[header_size] = '\0';

    cJSON* json = cJSON_Parse(json_string);

    //printf("%s\n",json_string);

    if (json==NULL){
        fprintf(stderr,"Error parsing JSON: %s\n", cJSON_GetErrorPtr());
        free(json_string);
        memory_unmap(data, size);
        exit(EXIT_FAILURE);
    }
    void* ptr = (uint8_t*)data + 8 + header_size;

    copy_to_f32(json,"lm_head.weight",ptr,lm_heads.data);
    copy_to_f32(json,"model.embed_tokens.weight",ptr,embeds.data);
    copy_to_f32(json,"model.norm.weight",ptr,output_norm.data);
    char name[128] = {0};
    for(int i=0;i<num_layers;++i) {
        sprintf(name,"model.layers.%d.input_layernorm.weight",i);
        copy_to_f32(json,name,ptr,layers[i].attn_norm.data);

        sprintf(name,"model.layers.%d.self_attn.k_proj.weight",i);
        copy_to_f32(json,name,ptr,layers[i].k_weight.data);

        sprintf(name,"model.layers.%d.self_attn.q_proj.weight",i);
        copy_to_f32(json,name,ptr,layers[i].q_weight.data);

        sprintf(name,"model.layers.%d.self_attn.o_proj.weight",i);
        copy_to_f32(json,name,ptr,layers[i].o_weight.data);

        sprintf(name,"model.layers.%d.self_attn.v_proj.weight",i);
        copy_to_f32(json,name,ptr,layers[i].v_weight.data);    
        
        sprintf(name,"model.layers.%d.self_attn.k_norm.weight",i);
        copy_to_f32(json,name,ptr,layers[i].k_norm.data);               

        sprintf(name,"model.layers.%d.self_attn.q_norm.weight",i);
        copy_to_f32(json,name,ptr,layers[i].q_norm.data);   
        
        sprintf(name,"model.layers.%d.post_attention_layernorm.weight",i);
        copy_to_f32(json,name,ptr,layers[i].ffn_norm.data);     

        sprintf(name,"model.layers.%d.mlp.up_proj.weight",i);
        copy_to_f32(json,name,ptr,layers[i].up_weight.data);  
        
        sprintf(name,"model.layers.%d.mlp.down_proj.weight",i);
        copy_to_f32(json,name,ptr,layers[i].down_weight.data);    

        sprintf(name,"model.layers.%d.mlp.gate_proj.weight",i);
        copy_to_f32(json,name,ptr,layers[i].gate_weight.data);
    }

    pack_hot_weights();

    cJSON_Delete(json);
    free(json_string);
    memory_unmap(data,size);
}


void RunTimeMemory::init(const ModelConfig* config, int len){
    max_prefill_len = len;
    TENSOR_DATA_TYPE data_type = TENSOR_DATA_TYPE_F32;
    size_t dtype_size = get_data_type_size(data_type);
    int kv_elements = config->num_layers * config->kv_heads * config->max_seq_len * config->head_size;
    size_t kv_dtype_size = get_data_type_size(TENSOR_DATA_TYPE_BF16);
    k_data = malloc(kv_elements * kv_dtype_size);
    v_data = malloc(kv_elements * kv_dtype_size);


    // temp buffer, it dynamic change in the forward process
    int mha_elements = (config->q_heads * config->head_size * max_prefill_len        //q project temp value
                        + 2 * config->kv_heads * config->head_size * max_prefill_len //k/v temp value before caching as bf16
                        + config->q_heads * config->max_seq_len                      //attention score per token
                        );

    int mlp_elements = 2 * max_prefill_len * config->ffn_size;                       //up and gate project temp value        

    int max_elements = (mha_elements > mlp_elements ? mha_elements : mlp_elements);

    int inout_elements = max_prefill_len * config->embed_size;                       //layer input and output
    
    ptr = (uint8_t*)malloc((max_elements + 2*inout_elements) * dtype_size);
    
    if (!k.data || !v.data || !ptr) {
        fprintf(stderr, "malloc failed!\n");
        exit(EXIT_FAILURE);
    }

    in.data =(void*)ptr;
    out.data = (void*)(ptr + inout_elements*dtype_size);
    ptr += 2 * inout_elements * dtype_size;
    offset = 0;
    seq_len_processed = 0;
    //in.dim = out.dim = 0;

    k_cache.data_type = v_cache.data_type = TENSOR_DATA_TYPE_BF16;
    k_cache.dim = v_cache.dim = 3;
    k_cache.shape[0] = v_cache.shape[0] = 0;
    k_cache.shape[1] = v_cache.shape[1] = config->kv_heads;
    k_cache.shape[2] = v_cache.shape[2] = config->head_size;

    //init q k v tensor shape
    k.data_type = v.data_type = q.data_type = data_type;
    k.dim = v.dim = q.dim = 2;
    k.shape[1] = v.shape[1] = config->kv_heads * config->head_size;
    q.shape[1] = config->q_heads * config->head_size;

    //init up and gate tensor shape
    up.data_type = gate.data_type = data_type;
    up.dim = gate.dim = 2;
    up.shape[1] = gate.shape[1] = config->ffn_size;
}

void* RunTimeMemory::allocate_kv(int li, int seq_len, const ModelConfig* config,bool is_k) {
    uint8_t* data = NULL;
    int dtype_size = get_data_type_size(k_cache.data_type);
    if(is_k) {
        data = (uint8_t*)k_data;
        k_cache.shape[0] = seq_len + seq_len_processed;
        k_cache.data = data + li * config->max_seq_len * config->kv_heads * config->head_size * dtype_size;
    }
    else{
        data = (uint8_t*)v_data;
        v_cache.shape[0] = seq_len + seq_len_processed;
        v_cache.data = data + li * config->max_seq_len * config->kv_heads * config->head_size * dtype_size;
    }

    uint8_t* res = data
                + li * config->max_seq_len * config->kv_heads * config->head_size * dtype_size
                + seq_len_processed * config->kv_heads * config->head_size * dtype_size; 
    return (void*)res;
}


void RunTimeMemory::destroy() {
    free(k_data);
    free(v_data);
    free(in.data);
}


void RunTimeMemory::reinit(const ModelConfig* config, int len) {
    if(len >= max_seq_len) {
        fprintf(stderr, "Error, input length exceed the max length!\n");
        exit(EXIT_FAILURE);
    }

    while(max_prefill_len < len) {
        max_prefill_len *= 2;
    }

    max_prefill_len = (max_prefill_len < max_seq_len ? max_prefill_len : max_seq_len);

    destroy();

    init(config, max_prefill_len);
}
