#!/usr/bin/env python

import os
from dotenv import load_dotenv
import chainlit as cl
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig
from typing import cast

load_dotenv()

@cl.on_chat_start
async def on_chat_start():
    api_key = os.getenv("OPENAI_APIKEY") 
    model = ChatOpenAI(model="gpt-4o", streaming=True, api_key=api_key)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You're a helpful assistant."),
            ("human", "{question}"),
        ]
    )
    runnable = prompt | model | StrOutputParser()
    cl.user_session.set("runnable", runnable)
    await cl.Message(content="こんにちは、何を話しましょうか？").send()

@cl.on_message
async def on_message(message: cl.Message):
    runnable = cast(Runnable, cl.user_session.get("runnable"))
    res = cl.Message(content="")
    async for chunk in runnable.astream(
        {"question": message.content},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await res.stream_token(chunk)
    
    await res.send()

    

if __name__ == '__main__':
    print('Hello, World!')