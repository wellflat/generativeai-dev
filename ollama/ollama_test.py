#!/usr/bin/env python

import asyncio

from langchain_ollama import ChatOllama


async def chat() -> None:
    chat = ChatOllama(
        model="phi4",
        temperature=0.5,
    )
    messages=[
        ("system", "You are a helpful assistant."),
        ("human", "なぜ空は青いのですか?"),
    ]
    async for chunk in chat.astream(messages):
        print(chunk.content)

if __name__ == "__main__":
    asyncio.run(chat())
