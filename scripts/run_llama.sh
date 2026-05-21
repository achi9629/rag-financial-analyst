export PATH="$HOME/.local/bin:$PATH"
export NCCL_P2P_DISABLE=1
CUDA_VISIBLE_DEVICES=0,1 vllm serve assets/model/Meta-Llama-3-70B-Instruct \
    --port 8000 \
    --tensor-parallel-size 2 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.9 \
    --quantization fp8