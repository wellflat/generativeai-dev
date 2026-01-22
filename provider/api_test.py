#!/usr/bin/env python

import os

from dotenv_flow import dotenv_flow
from openai import OpenAI

dotenv_flow("dev")

BASE_URL="https://api.groq.com/openai/v1"
#BASE_URL="https://openrouter.ai/api/v1"
api_key = os.environ.get("GROQ_API_KEY")
#api_key = os.environ.get("OPENROUTER_API_KEY")
client = OpenAI(base_url=BASE_URL, api_key=api_key)

MODEL_NAME = "openai/gpt-oss-120b"

def request_chat_completion_streaming(prompts: str) -> None:
    """
    Chat Completions APIをストリーミングモードで呼び出す
    生成されたトークンを逐次受け取る
    """
    print("\n--- Chat Completions ストリーミングリクエスト ---")

    try:
        stream = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "user", "content": prompts }
            ],
            max_tokens=2048,
            temperature=1.0,
            stream=True
        )

        print("生成されたテキスト:")
        for chunk in stream:
            # completions APIの .text ではなく、chat completions APIの .delta.content を使用
            content = chunk.choices[0].delta.content
            if content:
                print(content, end="", flush=True)
        print()

    except Exception as e:
        print(f"リクエスト中にエラーが発生しました: {e}")


if __name__ == "__main__":
    prompts = "仕事の熱意を取り戻すためのアイデアを5つ挙げてください"
    request_chat_completion_streaming(prompts)
