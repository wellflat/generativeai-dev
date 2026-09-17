#!/usr/bin/env python

import os
from openai import OpenAI

# vLLM APIサーバーのベースURL。環境変数から取得するか、デフォルトの localhost:8000 を使用します。
base_url = os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1")
# vLLMでは通常APIキーは不要ですが、ダミーの文字列を渡す必要があります。
api_key = os.environ.get("OPENAI_API_KEY", "token-not-needed")

client = OpenAI(
    base_url=base_url,
    api_key=api_key,
)

# デフォルトのモデル名（サーバーで別のモデルを起動している場合は動的に上書きされます）
model_name = "facebook/opt-125m"

def test_chat_completion():
    print("--- Chat Completion Test ---")
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello! Introduce yourself briefly in Japanese."}
            ],
            max_tokens=1024,  # Gemma 4の長い思考（Reasoning）プロセスを完了させるため、大きめの値を設定
            temperature=0.7,
        )
        message = response.choices[0].message
        
        # 思考（推論）プロセスの抽出
        reasoning = getattr(message, "reasoning", None) or message.model_extra.get("reasoning_content")
        if reasoning:
            print("--- Thinking Process ---")
            print(reasoning)
            print("------------------------")
            
        print("Response (Final Content):")
        print(message.content)
    except Exception as e:
        print(f"Error during chat completion: {e}")


if __name__ == "__main__":
    print(f"Connecting to vLLM server at: {base_url}")
    
    # 起動中のモデル一覧を取得し、テストに使用するモデル名を動的に決定する
    try:
        models = client.models.list()
        available_models = [model.id for model in models.data]
        print(f"Available models: {available_models}")
        if available_models:
            model_name = available_models[0]
            print(f"Using model: {model_name}\n")
    except Exception as e:
        print(f"Failed to fetch models from server: {e}")
        print(f"Falling back to default model name: {model_name}\n")

    test_chat_completion()
