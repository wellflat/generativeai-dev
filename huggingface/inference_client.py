#!/usr/bin/env python

import os

from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()

def chat(prompt: str) -> None:
    client = InferenceClient(
        base_url="https://router.huggingface.co/v1",
        api_key=os.environ["HF_TOKEN"],
    )

    completion = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role":"user", "content": prompt}]
    )

    print(completion.choices[0].message)


if __name__ == "__main__":
    chat("日本の首都はどこですか?")
