#!/bin/sh

MODEL=openai/openai/gpt-oss-120b:groq
TASK="../../.venv/lib/python3.12/site-packages/inspect_evals/aime2025/aime2025.py@aime2025"
#BASE_URL=http://localhost:30201/v1
BASE_URL=https://openrouter.ai/api/v1
#BASE_URL=https://api.groq.com/openai/v1
inspect eval ${TASK} --model-base-url ${BASE_URL} --model ${MODEL} --temperature=0.0 --top-p=1.0 --top-k=50
