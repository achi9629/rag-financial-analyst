export PATH="$HOME/.local/bin:$PATH"
CUDA_VISIBLE_DEVICES=2 vllm serve assets/model/Qwen2.5-Coder-32B-Instruct \
    --port 8001 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.9