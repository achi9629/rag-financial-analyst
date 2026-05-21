#!/bin/bash
export no_proxy="localhost,127.0.0.1"

PORT=${1:-8001}
MODEL="assets/model/Qwen2.5-Coder-32B-Instruct"

echo "Checking vLLM server on port $PORT..."

if curl -s --noproxy localhost http://localhost:$PORT/health > /dev/null 2>&1; then
    echo "✅ Server is healthy"
else
    echo "❌ Server not responding on port $PORT"
    exit 1
fi

echo "Sending test prompt..."
curl -s --noproxy localhost http://localhost:$PORT/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{\"model\": \"$MODEL\", \"messages\": [{\"role\": \"user\", \"content\": \"Write a pandas one-liner to count rows\"}], \"max_tokens\": 50, \"temperature\": 0}" | python -m json.tool

echo ""
echo "✅ Server ready at http://localhost:$PORT"