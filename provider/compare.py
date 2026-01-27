#!/usr/bin/env python

import asyncio
import os
import time

from dotenv_flow import dotenv_flow
from openai import AsyncOpenAI
from tabulate import tabulate

dotenv_flow("dev")


# APIキーの設定
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
CEREBRAS_API_KEY = os.environ.get("CEREBRAS_API_KEY")

# クライアントの初期化（OpenAI互換SDKを使用）
groq_client = AsyncOpenAI(base_url="https://api.groq.com/openai/v1", api_key=GROQ_API_KEY)
cerebras_client = AsyncOpenAI(base_url="https://api.cerebras.ai/v1", api_key=CEREBRAS_API_KEY)

async def fetch_ai_response(name, client, model, prompt) -> dict:
    print(f"[{name}] リクエスト開始...")
    start_time = time.perf_counter()

    # リクエスト実行
    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
        max_tokens=2048,
        stream=False # 統計取得のためストリームなし
    )

    end_time = time.perf_counter()
    duration = end_time - start_time

    # メトリクスの抽出
    usage = response.usage.model_dump()
    print(f"[{name}] Usage: {usage}")
    tokens = usage.get("completion_tokens", 0)

    # 固有の時間メトリクス取得 (秒単位)
    api_process_time = usage.get("total_time")

    # API側の時間が取得できた場合はそれを使用、なければクライアント計測時間(duration)を代用
    calc_base_time = api_process_time if api_process_time else duration

    tps = tokens / calc_base_time if calc_base_time > 0 else 0
    tps_client = tokens / duration if duration > 0 else 0

    return {
        "Provider": name,
        "Model": model,
        "Total Time (s)": round(duration, 3),
        "API Process (s)": round(api_process_time, 3) if api_process_time else "N/A",
        "Overhead (s)": round(duration - api_process_time, 3) if api_process_time else 0,
        "Tokens": tokens,
        "TPS (API)": round(tps, 2),
        "TPS (Client)": round(tps_client, 2)
    }

async def run_race() -> None:
    prompt = "「AI半導体の未来」について、500文字程度で論理的に解説してください。"

    tasks = [
        fetch_ai_response("Groq", groq_client, "openai/gpt-oss-120b", prompt),
        fetch_ai_response("Cerebras", cerebras_client, "gpt-oss-120b", prompt)
    ]

    print(f"\n🚀 レース開始！ プロンプト: {prompt[:20]}...\n")
    results = await asyncio.gather(*tasks)

    # 結果を表示
    print(tabulate(results, headers="keys", tablefmt="pretty"))

    # 勝者の判定
    winner_speed = min(results, key=lambda x: x["Total Time (s)"])
    winner_tps = max(results, key=lambda x: x["TPS (API)"])

    print(f"\n🏁 完走速度王者: {winner_speed['Provider']} ({winner_speed['Total Time (s)']}秒)")
    print(f"📈 スループット王者: {winner_tps['Provider']} ({winner_tps['TPS (API)']} tokens/sec)")

if __name__ == "__main__":
    asyncio.run(run_race())
