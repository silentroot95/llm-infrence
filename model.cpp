
#include <assert.h>
#include "cJSON.h"


#include "model.h"
#include "mmap.h"


void save_logits(const char* filename, float* logits, int m, int n) {
    FILE* f = fopen(filename, "wb");  // wb = binary write

    Header h = {m, n};
    fwrite(&h, sizeof(Header), 1, f);

    fwrite(logits, sizeof(float), m*n, f);
    fflush(f);
    fclose(f);
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


static void copy_data(const cJSON* json,const char* name,void* begin_ptr, Tensor* dst) {
    cJSON* tensor_info = cJSON_GetObjectItemCaseSensitive(json,name);
    cJSON* dtype = cJSON_GetObjectItemCaseSensitive(tensor_info,"dtype");
    cJSON* data_offsets = cJSON_GetObjectItemCaseSensitive(tensor_info, "data_offsets");

    assert(cJSON_IsArray(data_offsets));
    assert(cJSON_GetArraySize(data_offsets)==2);

    uint64_t offset = cJSON_GetArrayItem(data_offsets,0)->valuedouble;
    uint64_t end_offset = cJSON_GetArrayItem(data_offsets,1)->valuedouble;
    uint8_t* tensor_data = (uint8_t*)begin_ptr  + offset;

    uint64_t nbytes = end_offset - offset;
    TENSOR_DATA_TYPE data_type = get_data_type(dtype->valuestring);
    size_t num = nbytes / get_data_type_size(data_type);

    switch (dst->data_type){
    case TENSOR_DATA_TYPE_F32: {
        float* dst_data = (float*)dst->data;
        uint16_t* src = (uint16_t*)tensor_data; 
        size_t dst_num = dst->nelements();
        assert(num == dst_num);
        for(size_t i=0; i<num; ++i) {
            dst_data[i] = bf16_to_f32(src[i]);
        }
        break;
    }
    case TENSOR_DATA_TYPE_BF16:
        memcpy(dst->data,tensor_data,nbytes);
        break;
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
    size_t bf16_dtype_size = get_data_type_size(TENSOR_DATA_TYPE_BF16);

    size_t f32_elements = (
        2*config->embed_size * config->vocab_size  //embed and lm_head 
        + config->embed_size                      //final_norm
        + config->num_layers*(
            2*config->embed_size                                             //attntion norm and ffn norm
            + 2*config->q_heads * config->head_size * config->embed_size       //q weight and o weight
            + 2*config->kv_heads * config->head_size * config->embed_size      //k v weight
            + 2 * config->head_size                                          //q k norm
            + 3 * config->embed_size * config->ffn_size                       // down weight
        )
    );

    data = malloc(dtype_size * f32_elements );

    if(!data) {
        fprintf(stderr,"%s,%d malloc failed!\n", __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }
    
    layers = (TransformerLayer*)malloc(sizeof(TransformerLayer)*config->num_layers);
    num_layers = config->num_layers;
    if(!layers) {
        fprintf(stderr,"%s,%d: malloc failed!\n", __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }

    uint8_t* ptr = (uint8_t*)data;

    embeds.data_type = dtype;
    embeds.dim = 2;
    embeds.shape[0] = config->vocab_size;
    embeds.shape[1] = config->embed_size;
    embeds.data = (void*)ptr;
    ptr += embeds.nbytes();
    
    lm_heads.copy_meta(&embeds);
    lm_heads.data = (void*)ptr;
    ptr += lm_heads.nbytes();

    output_norm.data_type = dtype;
    output_norm.dim = 1;
    output_norm.shape[0] = config->embed_size;
    output_norm.data = (void*)ptr;
    ptr += output_norm.nbytes();

    for(int i=0;i<config->num_layers;++i) {
        layers[i].attn_norm.data_type = dtype;
        layers[i].attn_norm.dim = 1;
        layers[i].attn_norm.shape[0] = config->embed_size;
        layers[i].attn_norm.data = (void*)ptr;
        ptr += layers[i].attn_norm.nbytes();

        layers[i].q_weight.data_type = dtype;
        layers[i].q_weight.dim = 2;
        layers[i].q_weight.shape[0] = config->q_heads * config->head_size;
        layers[i].q_weight.shape[1] = config->embed_size;
        layers[i].q_weight.data = (void*)ptr;
        ptr += layers[i].q_weight.nbytes();

        layers[i].k_weight.data_type = dtype;
        layers[i].k_weight.dim = 2;
        layers[i].k_weight.shape[0] = config->kv_heads * config->head_size;
        layers[i].k_weight.shape[1] = config->embed_size;
        layers[i].k_weight.data = (void*)ptr;
        ptr += layers[i].k_weight.nbytes();

        layers[i].v_weight.copy_meta(&layers[i].k_weight);
        layers[i].v_weight.data = (void*)ptr;
        ptr += layers[i].v_weight.nbytes();

        layers[i].o_weight.data_type = dtype;
        layers[i].o_weight.dim = 2;
        layers[i].o_weight.shape[0] = config->embed_size;
        layers[i].o_weight.shape[1] = config->q_heads * config->head_size;
        layers[i].o_weight.data = (void*)ptr;
        ptr += layers[i].o_weight.nbytes();

        layers[i].q_norm.data_type = dtype;
        layers[i].q_norm.dim = 1;
        layers[i].q_norm.shape[0] = config->head_size;
        layers[i].q_norm.data = (void*)ptr;
        ptr += layers[i].q_norm.nbytes();

        layers[i].k_norm.copy_meta(&layers[i].q_norm);
        layers[i].k_norm.data = (void*)ptr;
        ptr += layers[i].k_norm.nbytes();

        layers[i].ffn_norm.copy_meta(&layers[i].attn_norm);
        layers[i].ffn_norm.data = (void*)ptr;
        ptr += layers[i].ffn_norm.nbytes();
        
        //up and gate weight use bf16 to improve decode speed
        layers[i].up_weight.data_type = dtype;
        layers[i].up_weight.dim = 2;
        layers[i].up_weight.shape[0] = config->ffn_size;
        layers[i].up_weight.shape[1] = config->embed_size;
        layers[i].up_weight.data = (void*)ptr;
        ptr += layers[i].up_weight.nbytes();

        layers[i].gate_weight.copy_meta(&layers[i].up_weight);
        layers[i].gate_weight.data = (void*)ptr;
        ptr += layers[i].gate_weight.nbytes();

        layers[i].down_weight.data_type = dtype;
        layers[i].down_weight.dim = 2;
        layers[i].down_weight.shape[0] = config->embed_size;
        layers[i].down_weight.shape[1] = config->ffn_size;
        layers[i].down_weight.data = (void*)ptr;
        ptr += layers[i].down_weight.nbytes();
    }
    ptr = nullptr;
}

void ModelWeights::load_tensor(const char* path) {
    size_t size = 0;
    void* data = memory_map(path, &size);

    uint64_t header_size = *((uint64_t*)data);

    char* json_string = (char*)malloc(header_size+1);
    memcpy(json_string, (char*)data+8, header_size);
    json_string[header_size] = '\0';

    cJSON* json = cJSON_Parse(json_string);

    if (json==NULL){
        fprintf(stderr,"Error parsing JSON: %s\n", cJSON_GetErrorPtr());
        free(json_string);
        memory_unmap(data, size);
        exit(EXIT_FAILURE);
    }
    void* ptr = (uint8_t*)data + 8 + header_size;

    copy_data(json,"model.embed_tokens.weight",ptr,&embeds);
    copy_data(json,"lm_head.weight",ptr,&lm_heads);
    copy_data(json,"model.norm.weight",ptr,&output_norm);

    char name[128] = {0};
    for(int i=0;i<num_layers;++i) {
        sprintf(name,"model.layers.%d.input_layernorm.weight",i);
        copy_data(json,name,ptr,&layers[i].attn_norm);

        sprintf(name,"model.layers.%d.self_attn.q_proj.weight",i);
        copy_data(json,name,ptr,&layers[i].q_weight);

        sprintf(name,"model.layers.%d.self_attn.k_proj.weight",i);
        copy_data(json,name,ptr,&layers[i].k_weight);

        sprintf(name,"model.layers.%d.self_attn.v_proj.weight",i);
        copy_data(json,name,ptr,&layers[i].v_weight);    

        sprintf(name,"model.layers.%d.self_attn.o_proj.weight",i);
        copy_data(json,name,ptr,&layers[i].o_weight);

        sprintf(name,"model.layers.%d.self_attn.q_norm.weight",i);
        copy_data(json,name,ptr,&layers[i].q_norm);   

        sprintf(name,"model.layers.%d.self_attn.k_norm.weight",i);
        copy_data(json,name,ptr,&layers[i].k_norm);               
        
        sprintf(name,"model.layers.%d.post_attention_layernorm.weight",i);
        copy_data(json,name,ptr,&layers[i].ffn_norm);     

        sprintf(name,"model.layers.%d.mlp.up_proj.weight",i);
        copy_data(json,name,ptr,&layers[i].up_weight);  
       
        sprintf(name,"model.layers.%d.mlp.gate_proj.weight",i);
        copy_data(json,name,ptr,&layers[i].gate_weight);
        
        sprintf(name,"model.layers.%d.mlp.down_proj.weight",i);
        copy_data(json,name,ptr,&layers[i].down_weight);    
    }

    ptr = nullptr;

    cJSON_Delete(json);
    free(json_string);
    memory_unmap(data,size);
}

inline void ModelWeights::destory() {
    free(layers);
    free(data);
}


static inline size_t align_size(size_t size, int align=64) {
    return (size + align - 1) & ~(align - 1);
}

void RunTimeMemory::init(const ModelConfig* config, int len){
    m_prefill_chunck = len;
    TENSOR_DATA_TYPE data_type = TENSOR_DATA_TYPE_F32;
    size_t dtype_size = get_data_type_size(data_type);
    int kv_elements = config->num_layers * config->kv_heads * config->max_seq_len * config->head_size;
    size_t kv_dtype_size = get_data_type_size(TENSOR_DATA_TYPE_F32);
    k_data = malloc(kv_elements * kv_dtype_size);
    v_data = malloc(kv_elements * kv_dtype_size);


    // temp buffer, it dynamic change in the forward process
    size_t mha_size = align_size(config->q_heads * config->head_size * m_prefill_chunck * dtype_size, 64); //q project temp value
        
    mha_size += align_size(config->q_heads * config->max_seq_len * dtype_size, 64);  //attention score per token

    size_t mlp_size = 2 * align_size(m_prefill_chunck * config->ffn_size * dtype_size, 64);  //up and gate project temp value        

    mlp_size += align_size(m_prefill_chunck * config->ffn_size * dtype_size, 64);  // for debug sigmoid(gate) value;

    size_t max_size = (mha_size > mlp_size ? mha_size : mlp_size);

    size_t final_logit_elements = align_size(m_prefill_chunck * config->vocab_size * dtype_size, 64);  //final logit before softmax

    max_size = (max_size > final_logit_elements ? max_size : final_logit_elements);
    
    size_t inout_size = align_size(m_prefill_chunck * config->embed_size * dtype_size, 64);  //layer input and output
    
    size_t total_size = 2 * inout_size + max_size; // in and out + temp buffer

    ptr = (uint8_t*)malloc(total_size);
    
    if (!k_data || !v_data || !ptr) {
        fprintf(stderr,"%s,%d malloc failed!\n", __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }

    in.data =(void*)ptr;
    out.data = (void*)(ptr + inout_size);

    ptr += 2 * inout_size;
    capacity = total_size - 2 * inout_size;
    offset = 0;

    seq_len_processed = 0;
    //in.dim = out.dim = 0;

    k_cache.data_type = v_cache.data_type = TENSOR_DATA_TYPE_F32;
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
    //uint8_t* data = NULL;
    //int dtype_size = get_data_type_size(k_cache.data_type);
    float* data = nullptr;
    if(is_k) {
        data = (float*)k_data;
        k_cache.shape[0] = seq_len + seq_len_processed;
        k_cache.data = data + li * config->max_seq_len * config->kv_heads * config->head_size;
    }
    else{
        data = (float*)v_data;
        v_cache.shape[0] = seq_len + seq_len_processed;
        v_cache.data = data + li * config->max_seq_len * config->kv_heads * config->head_size;
    }

    data += li * config->max_seq_len * config->kv_heads * config->head_size
                + seq_len_processed * config->kv_heads * config->head_size; 
    return (void*)data;
}

void RunTimeMemory::destroy() {
    free(k_data);
    free(v_data);
    free(in.data);
}
