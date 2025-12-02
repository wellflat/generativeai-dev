#!/usr/bin/env python
import os

from openai import OpenAI

BASE_URL="https://openrouter.ai/api/v1"
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(base_url=BASE_URL, api_key=api_key)

MODEL_NAME = "openai/gpt-oss-120b"

def request_chat_completion_non_streaming(prompts: str):
    """
    Chat Completions APIを非ストリーミングモードで呼び出す
    レスポンス全体を一度に受け取る
    """
    print("--- Chat Completions 非ストリーミングリクエスト ---")

    try:
        chat_completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "user", "content": prompts}
            ],
            max_tokens=2048,
            temperature=1.0,
            stream=False
        )

        # 生成されたテキストを表示
        print("生成されたテキスト:")
        print(chat_completion.choices[0].message.content)

        # トークン使用量を表示
        print("\nトークン使用量:")
        print(chat_completion.usage)
    except Exception as e:
        print(f"リクエスト中にエラーが発生しました: {e}")

def request_chat_completion_streaming(prompts: str):
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
    request_chat_completion_non_streaming(prompts)
    request_chat_completion_streaming(prompts)
