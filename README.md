# LLM Inference

一个基于 C++ 实现的轻量级大语言模型推理项目，当前示例代码面向 `Qwen3-0.6B` 模型，完成了从配置读取、权重加载、Tokenizer 编码、Chat Template 拼接，到 Prefill / Decode 推理和性能统计的完整流程。

## 项目实现

- RunTimeMemory 统一内存管理，无额外运行时开销，支持chunck prefill
- 支持AVX2和FMA加速矩阵计算
- 部分权重INT8量化，提升 decode 阶段的性能
- BPE 分词实现和 jinja Chat 模板的应用
- 支持 temperature, top k, top p 的输出采样
- 支持 prefill / decode 阶段的吞吐和算子级性能统计


## 项目结构

```text
.
├── main.cpp         # 程序入口，负责加载模型并执行推理
├── model.h/.cpp     # 模型配置、权重加载、运行时内存管理
├── operator.h/.cpp  # 核心算子，如 matmul rms_norm attention
├── tokenizer.h/.cpp # BPE 分词、聊天模板、采样器
├── tensor.h         # Tensor 结构与数据类型定义
├── mmap.h           # 跨平台内存映射读取
├── qwen3.h          # Qwen3 前向推理封装与性能profile
├── cJSON.h/.c       # 第三方依赖，解析JSON文件
└── chat.jinja       # 聊天模板参考文件
```

## 模型文件准备

从huggingface下载[Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)模型文件，需要文件列表如下：

```text
./qwen3-0.6b/
├── config.json
├── model.safetensors
├── vocab.json
├── merges.txt
├── tokenizer_config.json
└── generation_config.json
```

## 依赖要求

- Linux / macOS 环境优先
- `g++`，建议支持 C++17
- OpenMP
- 支持 AVX2/FMA 的 CPU 可获得更好的性能


## 编译

```bash
g++ -std=c++17 -g -O3 -mavx2 -mfma -ffast-math -fopenmp \
  main.cpp model.cpp tokenizer.cpp operator.cpp cJSON.c \
  -o llm_inference
```

如果你的平台不支持 `avx2`和`fma`，可以去掉`-mavx2 -mfma`编译选项。

## 运行

编译完成后直接执行：

```bash
./llm_inference
```

## 采样与生成设置

当前示例中的采样器参数定义在 [main.cpp](/home/wwb/llm-infrence/main.cpp)：

```cpp
Sampler sampler(model_config.vocab_size, 0.6, 0.95, 20);
```

对应含义：

- `temperature = 0.6`
- `top_p = 0.95`
- `top_k = 20`

