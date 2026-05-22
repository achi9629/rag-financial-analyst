# vLLM Serve Parameters Reference

## Common Parameters

| Parameter | Default | Description |
|---|---|---|
| `--model` | (required) | Path or HuggingFace model ID |
| `--port` | 8000 | Port to serve on |
| `--host` | 0.0.0.0 | Host to bind to |
| `--max-model-len` | auto | Max sequence length (context window). Lower = less VRAM |
| `--gpu-memory-utilization` | 0.9 | Fraction of GPU memory to use (0.0–1.0) |
| `--tensor-parallel-size` | 1 | Number of GPUs for tensor parallelism |
| `--dtype` | auto | Data type: `auto`, `float16`, `bfloat16`, `float32` |
| `--quantization` | None | Quantization method: `awq`, `gptq`, `squeezellm`, `fp8` |
| `--max-num-seqs` | 256 | Max sequences processed in parallel (batch size) |
| `--max-num-batched-tokens` | auto | Max tokens processed per batch |

## GPU / Memory

| Parameter | Default | Description |
|---|---|---|
| `--gpu-memory-utilization` | 0.9 | 0.5 = use 50% of VRAM. Lower leaves room for other processes |
| `--swap-space` | 4 | CPU swap space (GB) for KV cache overflow |
| `--kv-cache-dtype` | auto | KV cache dtype: `auto`, `fp8`, `fp8_e5m2`, `fp8_e4m3` |
| `--block-size` | 16 | Token block size for paged attention |
| `--enforce-eager` | False | Disable CUDA graph (slower but saves memory) |

## Serving / API

| Parameter | Default | Description |
|---|---|---|
| `--served-model-name` | model path | Name exposed in the API (for `/v1/models`) |
| `--api-key` | None | API key to require for requests |
| `--chat-template` | auto | Jinja2 chat template path or string |
| `--response-role` | assistant | Role name in chat completions |
| `--trust-remote-code` | False | Allow executing model's custom code |

## Performance Tuning

| Parameter | Default | Description |
|---|---|---|
| `--disable-log-requests` | False | Don't log individual requests (faster) |
| `--disable-log-stats` | False | Don't log stats periodically |
| `--max-log-len` | None | Truncate logged prompts to this length |
| `--seed` | 0 | Random seed for reproducibility |
| `--enable-prefix-caching` | False | Cache common prompt prefixes (good for RAG) |

## Parallelism

| Parameter | Default | Description |
|---|---|---|
| `--tensor-parallel-size` | 1 | Split model layers across N GPUs |
| `--pipeline-parallel-size` | 1 | Pipeline parallelism (less common) |
| `--distributed-executor-backend` | ray | Backend: `ray` or `mp` (multiprocessing) |

---

## Our Project Commands

```bash
# Qwen-only (today — single GPU)
CUDA_VISIBLE_DEVICES=0 vllm serve assets/model/Qwen2.5-Coder-32B-Instruct \
    --port 8000 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.9

# Llama-3-70B (when available — 2 GPUs)
vllm serve meta-llama/Meta-Llama-3-70B-Instruct \
    --port 8000 \
    --tensor-parallel-size 2 \
    --max-model-len 4096

# Qwen on second port (when running both)
CUDA_VISIBLE_DEVICES=2 vllm serve Qwen/Qwen2.5-Coder-32B-Instruct \
    --port 8001 \
    --max-model-len 4096
```

## Memory Estimation

| Model | FP16 VRAM | With `--gpu-memory-utilization` |
|---|---|---|
| Qwen2.5-Coder-32B | ~65 GB | Needs >=0.85 on 80GB A100 |
| Llama-3-70B (TP=2) | ~140 GB | ~70GB/GPU, needs >=0.9 per GPU |
| all-MiniLM-L6-v2 | <1 GB | CPU is fine |

## Tips

- `--max-model-len 4096` saves significant KV cache memory vs default (often 32k+)
- `--enable-prefix-caching` helps when many queries share the same system prompt
- `--enforce-eager` if you hit CUDA graph memory issues (trades speed for memory)
- Use `--quantization awq` to fit larger models on fewer GPUs (~40% VRAM reduction)
- `CUDA_VISIBLE_DEVICES=X` controls which GPU(s) the process sees
