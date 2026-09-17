#!/bin/sh

MODEL=mlx-community/gemma-4-e4b-it-4bit

uv tool run vllm-mlx serve ${MODEL} \
  --reasoning-parser gemma4 \
  --enable-auto-tool-choice \
  --tool-call-parser gemma4 \
  --default-chat-template-kwargs '{"enable_thinking": true}'
