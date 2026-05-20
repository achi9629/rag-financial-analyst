export PATH="$HOME/.local/bin:$PATH"
CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve assets/model/Qwen2.5-Coder-32B-Instruct \
    --port 8000 \
    --max-model-len 4096 \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.5