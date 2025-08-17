# llama.cpp

## **1. 完整克隆仓库（关键！）**

```bash
git clone https://github.com/ggerganov/llama.cpp

- 
- 不要用 `--depth=1`（浅克隆会导致 CMake 找不到版本信息）。
- 如果网络慢，可以用 GitHub 镜像站（如 `https://ghproxy.com/https://github.com/...`）


cd llama.cpp
git submodule update --init --recursive  # 初始化子模块

# 1. 生成构建配置（启用 CUDA）
cmake -B build -DGGML_CUDA=ON
```



>可能遇到的错误:
>CMake Error at common/CMakeLists.txt:85 (message):
>Could NOT find CURL.  Hint: to disable this feature, set -DLLAMA_CURL=OFF

Ubuntu/Debian

```bash
sudo apt update && sudo apt install -y libcurl4-openssl-dev
```

CentOS/RHEL

```bash
sudo yum install -y libcurl-devel
```

macOS（Homebrew）

```bash
brew install curl
```

然后重新运行 CMake：

```bash
2. 开始编译（Release 模式）

cmake --build build --config Release
```



## 2. BUG

```python
'''指定导出的名字'''
python convert_hf_to_gguf.py --outfile ./export_models/test.gguf ../LLaMA-Factory-main/models/DeepSeek-R1-0528-Qwen3-8B/

'''不指定，默认为源文件夹-F16.gguf  eg：DeepSeek-R1-0528-Qwen3-8B-F16.gguf'''
python convert_hf_to_gguf.py --outfile ./export_models/ ../LLaMA-Factory-main/models/DeepSeek-R1-0528-Qwen3-8B/

''' --outtype 指定输出类型
q2_k：特定张量（Tensor）采用较高的精度设置，而其他的则保持基础级别。
q3_k_l、q3_k_m、q3_k_s：这些变体在不同张量上使用不同级别的精度，从而达到性能和效率的平衡。
q4_0：这是最初的量化方案，使用 4 位精度。
q4_1 和 q4_k_m、q4_k_s：这些提供了不同程度的准确性和推理速度，适合需要平衡资源使用的场景。
q5_0、q5_1、q5_k_m、q5_k_s：这些版本在保证更高准确度的同时，会使用更多的资源并且推理速度较慢。
q6_k 和 q8_0：这些提供了最高的精度，但是因为高资源消耗和慢速度，可能不适合所有用户。
fp16 和 f32: 不量化，保留原始精度。

通用场景：直接选 Q4_K_M（4-bit 最佳平衡）。
显存紧张：
优先 Q3_K_M（3-bit）或 IQ2_XXS（2.5-bit）。
高精度需求：
选 Q5_K_M（5-bit）或 Q6_K（6-bit）。
'''

python convert_hf_to_gguf.py --outfile ./export_models/ ../LLaMA-Factory-main/models/DeepSeek-R1-0528-Qwen3-8B/  --outtype q8_0


```

>当你试图转换DeepSeek-R1-0528-Qwen3-8B 模型时遇到如下bug
>
>BPE pre-tokenizer was not recognized - update get_vocab_base_pre()

```python
#修改convert_hf_to_gguf_update 文件

'''
此外你需要确保
models =[ {"name": "llama-spm", "tokt": TOKENIZER_TYPE.SPM, "repo": "https://huggingface.co/meta-llama/Llama-2-7b-hf", }...]里面对应的huggingface模型你有权限下载，否则convert_hf_to_gguf_update.py会运行失败，如果模型拥有者不同意你的申请，可以先删除掉再运行convert_hf_to_gguf_update.py

我们可以先把其它的全部删除
https://huggingface.co  替换为  https://hf-mirror.com
'''
models = [
    {"name": "deepseek-r1-qwen3", "tokt": TOKENIZER_TYPE.BPE, "repo": "https://huggingface.co/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B", },
]
```

```bash
python convert_hf_to_gguf_update.py
```

```python
''' convert_hf_to_gguf.py 中将出现如下内容'''
if chkhsh == "e636dc30a262dcc0d8c323492e32ae2b70728f4df7dfe9737d9f920a282b8aea":
    # ref: https://hf-mirror.com//deepseek-ai/DeepSeek-R1-0528-Qwen3-8B
    res = "deepseek-r1-qwen3"
```

有可能依旧会报错，错误如下：

此时，我们可以复制报错信息中chkhsh的值，去替换convert_hf_to_gguf.py 中对应的chkhsh的值

```bash
WARNING:hf-to-gguf:**************************************************************************************
WARNING:hf-to-gguf:** WARNING: The BPE pre-tokenizer was not recognized!
WARNING:hf-to-gguf:**          There are 2 possible reasons for this:
WARNING:hf-to-gguf:**          - the model has not been added to convert_hf_to_gguf_update.py yet
WARNING:hf-to-gguf:**          - the pre-tokenization config has changed upstream
WARNING:hf-to-gguf:**          Check your model files and convert_hf_to_gguf_update.py and update them accordingly.
WARNING:hf-to-gguf:** ref:     https://github.com/ggml-org/llama.cpp/pull/6920
WARNING:hf-to-gguf:**
WARNING:hf-to-gguf:** chkhsh:  b0f33aec525001c9de427a8f9958d1c8a3956f476bec64403680521281c032e2
WARNING:hf-to-gguf:**************************************************************************************
WARNING:hf-to-gguf:
```



```zsh
#echo "系统有 $(nproc) 个 CPU 核心"
nproc
#如果想查看 CPU 型号、架构等完整信息，可以用：
lscpu
# 查看linux系统信息
cat /etc/os-release
```

