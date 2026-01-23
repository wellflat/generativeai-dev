#!/usr/bin/env python

import os
import argparse
import sys
import time

from dotenv_flow import dotenv_flow
from openai import OpenAI

dotenv_flow("dev")

PROVIDERS = {
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "api_key_env": "GROQ_API_KEY",
        "default_model": "llama-3.3-70b-versatile"
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        "default_model": "meta-llama/llama-3-8b-instruct:free"
    },
    "cerebras": {
        "base_url": "https://api.cerebras.ai/v1",
        "api_key_env": "CEREBRAS_API_KEY",
        "default_model": "llama3.1-8b"
    }
}

def request_chat_completion_streaming(client: OpenAI, model_name: str, prompts: str) -> None:
    """
    Chat Completions APIをストリーミングモードで呼び出す
    生成されたトークンを逐次受け取る
    """
    print("\n--- Chat Completions ストリーミングリクエスト ---")

    start_time = time.time()
    first_token_time = None
    token_count = 0

    try:
        stream = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": prompts }
            ],
            max_tokens=2048,
            temperature=0.0,
            stream=True
        )

        print("生成されたテキスト:")
        for chunk in stream:
            if first_token_time is None:
                first_token_time = time.time()

            # completions APIの .text ではなく、chat completions APIの .delta.content を使用
            content = chunk.choices[0].delta.content
            if content:
                print(content, end="", flush=True)
                token_count += 1
        print()

        end_time = time.time()

        if first_token_time:
            ttft = (first_token_time - start_time) * 1000
            total_latency = (end_time - start_time) * 1000
            generation_time = end_time - first_token_time

            print("\n" + "="*40)
            print(f"📊 Performance Metrics")
            print("="*40)
            print(f"TTFT (Time To First Token): {ttft:.2f} ms")
            print(f"Total Latency             : {total_latency:.2f} ms")
            if generation_time > 0:
                print(f"Throughput (estimated)    : {token_count / generation_time:.2f} tokens/sec")
            print("="*40)

    except Exception as e:
        print(f"リクエスト中にエラーが発生しました: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM Provider API Test Script")
    parser.add_argument("--provider", "-p", type=str, choices=PROVIDERS.keys(), default="cerebras", help="Select the LLM provider")
    parser.add_argument("--model", "-m", type=str, help="Override the default model name")

    args = parser.parse_args()

    config = PROVIDERS[args.provider]
    api_key = os.environ.get(config["api_key_env"])

    if not api_key:
        print(f"Error: Environment variable {config['api_key_env']} is not set.")
        sys.exit(1)

    client = OpenAI(base_url=config["base_url"], api_key=api_key)
    model_name = args.model if args.model else config["default_model"]

    print(f"Testing Provider: {args.provider}")
    print(f"Target Model: {model_name}")

    prompts = "仕事の熱意を取り戻すためのアイデアを5つ挙げてください"
    request_chat_completion_streaming(client, model_name, prompts)
