# LLM Inference

一个基于 C++ 实现的轻量级大语言模型推理项目，当前示例代码面向 `Qwen3-0.6B` 模型，完成了从配置读取、权重加载、Tokenizer 编码、Chat Template 应用，到 Prefill / Decode 推理和性能统计的完整流程。

## 项目实现

- RunTimeMemory 统一内存管理，无额外运行时开销，支持chunck prefill
- 支持AVX2和FMA加速矩阵计算
- 部分权重INT8量化，提升 decode 阶段的性能
- 支持 BPE 分词和 jinja Chat 模板的应用
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

- Linux / macOS / Windows
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
## 性能

>**prefill**

[Perf] prefill: 147 tokens, 1151.61 ms, 127.65 tokens/s<br>
[Profile] prefill total op time: 1151.54 ms<br>
[Profile] embed          0.45 ms    0.04%  calls=1<br>
[Profile] rms_norm       7.80 ms    0.68%  calls=113<br>
[Profile] matmul_qkv   246.86 ms   21.44%  calls=84<br>
[Profile] matmul_o     110.27 ms    9.58%  calls=28<br>
[Profile] matmul_ffn_up_gate   496.51 ms   43.12%  calls=28<br>
[Profile] matmul_ffn_down   185.62 ms   16.12%  calls=28<br>
[Profile] matmul_lm_head     3.57 ms    0.31%  calls=1<br>
[Profile] rope          68.69 ms    5.97%  calls=56<br>
[Profile] attention     24.81 ms    2.15%  calls=28<br>
[Profile] add            6.96 ms    0.60%  calls=56<br>

>**decode**

[Perf] decode : 661 tokens, 36161.25 ms, 18.28 tokens/s<br>
[Profile] decode total op time: 36135.46 ms<br>
[Profile] embed          1.00 ms    0.00%  calls=661<br>
[Profile] rms_norm      78.67 ms    0.22%  calls=74693<br>
[Profile] matmul_qkv  8252.82 ms   22.84%  calls=55524<br>
[Profile] matmul_o    3844.32 ms   10.64%  calls=18508<br>
[Profile] matmul_ffn_up_gate 11318.65 ms   31.32%  calls=18508<br>
[Profile] matmul_ffn_down  5694.85 ms   15.76%  calls=18508<br>
[Profile] matmul_lm_head  2687.64 ms    7.44%  calls=661<br>
[Profile] rope         423.20 ms    1.17%  calls=37016<br>
[Profile] attention   3784.74 ms   10.47%  calls=18508<br>
[Profile] add           49.59 ms    0.14%  calls=37016<br>