#!/usr/bin/env python

import httpx

API_URL="http://localhost:11434/api/chat"

def run() -> None:
  headers = {"Content-type": "application/json"}
  json = {
    "model": "gemma",
    "messages": [
      { "role": "user", "content": "hello" }
    ]
  }
  response = httpx.post(API_URL, json=json, headers=headers, timeout=30)
  print(response.content)

if __name__ == "__main__":
  run()
