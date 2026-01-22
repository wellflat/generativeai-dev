#!/usr/bin/env python

import os
from datetime import datetime

from openai import OpenAI

# 1. クライアントの初期化
# 環境変数に GROQ_API_KEY を設定していることを前提
client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ.get("GROQ_API_KEY")
)

def run_benchmarking_query(prompt, model="llama-3.3-70b-versatile"):
    print(f"--- Querying Model: {model} ---")

    # 2. リクエストの実行
    # service_tier="performance" を追加して最速設定にすることも可能
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        service_tier="on_demand" 
    )

    # 3. メトリクスの抽出
    usage = response.usage
    # Groq固有のメトリクスは usage オブジェクト内に直接、または辞書形式で入っている
    # OpenAI SDK経由の場合、属性としてアクセスできるか、extra_fieldsを確認

    # 辞書形式に変換して安全に取得
    usage_dict = usage.model_dump()

    prompt_tokens = usage_dict.get("prompt_tokens", 0)
    completion_tokens = usage_dict.get("completion_tokens", 0)

    # Groq独自の実行時間（秒からミリ秒に変換）
    q_time = usage_dict.get("prompt_time", 0) * 1000      # 最初のトークンまでの準備
    c_time = usage_dict.get("completion_time", 0) * 1000  # 生成時間
    total_t = usage_dict.get("total_time", 0) * 1000     # 合計
    queue_t = usage_dict.get("queue_time", 0) * 1000     # 待ち時間

    # 4. 結果の表示
    print(f"Result: {response.choices[0].message.content[:50]}...")
    print("\n" + "="*40)
    print(f"🚀 GROQ PERFORMANCE METRICS")
    print("="*40)
    print(f"Tokens Used      : {prompt_tokens} (Prompt) + {completion_tokens} (Completion)")
    print(f"Total Tokens     : {usage_dict.get('total_tokens')}")
    print("-"*40)
    print(f"Queue Time       : {queue_t:>8.2f} ms (待ち時間)")
    print(f"Prompt Time      : {q_time:>8.2f} ms (読込時間)")
    print(f"Completion Time  : {c_time:>8.2f} ms (生成時間)")
    print(f"Total Latency    : {total_t:>8.2f} ms")
    print("-"*40)

    # スループット（トークン/秒）の計算
    if c_time > 0:
        tps = completion_tokens / (c_time / 1000)
        print(f"Throughput       : {tps:>8.2f} tokens/sec")
    print("="*40 + "\n")

run_benchmarking_query("AI半導体の未来について100文字程度で簡潔に教えてください。")
