#!/usr/bin/env python

from langchain_ollama import ChatOllama


def chat():
    chat = ChatOllama(
        model="gemma",
        temperature=0.5,
    )
    messages=[
        ("system", "You are a helpful assistant."),
        ("human", "Hello!"),
    ]
    for chunk in chat.stream(messages):
        print(chunk)
    
if __name__ == "__main__":
    chat()
    