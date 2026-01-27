#!/usr/bin/env python

import os
import time
from cerebras.cloud.sdk import Cerebras
from dotenv_flow import dotenv_flow

dotenv_flow("dev")


# 1. クライアントの初期化
client = Cerebras(
    api_key=os.environ.get("CEREBRAS_API_KEY")
)

def run_cerebras_benchmark(prompt, model="gpt-oss-120b"):
    print(f"--- Cerebras Inference: {model} ---")

    # 開始時間を記録（TTFT計算用）
    start_time = time.perf_counter()

    # 2. リクエストの実行
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=model,
    )
    end_time = time.perf_counter()

    # 3. メトリクスの抽出
    # Cerebrasは response.usage に詳細な統計を格納しています
    usage = response.usage

    # 基本トークン情報
    prompt_tokens = usage.prompt_tokens
    completion_tokens = usage.completion_tokens

    # キャッシュ情報 (2026年新機能: prompt_tokens_details内)
    cached_tokens = getattr(usage.prompt_tokens_details, 'cached_tokens', 0) if hasattr(usage, 'prompt_tokens_details') else 0

    # 時間指標（Cerebras Cloud APIは秒単位で提供）
    # 注意: フィールド名はプロバイダにより微差があるため、辞書で安全に取得
    u_dict = usage.model_dump()
    total_time = u_dict.get("total_time")
    if not total_time:
        total_time = end_time - start_time

    # 4. 結果の表示
    #print(f"\nResult: {response.choices[0].message.content[:50]}...")
    print(f"\nResult: {response.choices[0].message.content}...")
    print("\n" + "═"*45)
    print(f"🌀 CEREBRAS WSE-3 PERFORMANCE METRICS")
    print("═"*45)
    print(f"Model ID         : {model}")
    print(f"Tokens           : {prompt_tokens} (Prompt) / {completion_tokens} (Completion)")

    if cached_tokens > 0:
        print(f"Prompt Cache     : 🟢 HIT ({cached_tokens} tokens)")
    else:
        print("Prompt Cache     : ⚪ MISS")

    print("─"*45)
    # Cerebrasは極めて高いスループット（Tokens Per Second）が特徴
    if total_time > 0 and completion_tokens > 0:
        tps = completion_tokens / total_time
        print(f"Throughput       : {tps:>10.2f} tokens/sec")

    print(f"Total Latency    : {total_time * 1000:>10.2f} ms")
    print("═"*45 + "\n")

# 実行
run_cerebras_benchmark("大規模言語モデルの推論におけるウェハスケールエンジンの利点を教えてください。")
