#!/usr/bin/env python

import os
from dotenv import load_dotenv
import chainlit as cl
from chainlit.input_widget import Slider
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig
from typing import cast

load_dotenv()

@cl.set_chat_profiles
async def chat_profile():
    return [
        cl.ChatProfile(
            name="GPT-3.5",
            markdown_description="The underlying LLM model is **GPT-3.5**.",
            icon="https://picsum.photos/200",
        ),
        cl.ChatProfile(
            name="GPT-4",
            markdown_description="The underlying LLM model is **GPT-4**.",
            icon="https://picsum.photos/250",
        ),
    ]

@cl.on_chat_start
async def on_chat_start():
    settings = await cl.ChatSettings(
        [
            Slider(
                id="Temperature",
                label="OpenAI - Temperature",
                initial=1,
                min=0,
                max=2,
                step=0.1,
            ),
        ]
    ).send()
    chat_profile = cl.user_session.get("chat_profile")
    print(chat_profile)
    api_key = os.getenv("OPENAI_API_KEY")
    model = ChatOpenAI(
        model="gpt-4o",
        streaming=True,
        temperature=settings["Temperature"],
        api_key=api_key
    )
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
    print(cl.chat_context.to_openai())
    
    runnable = cast(Runnable, cl.user_session.get("runnable"))
    res = cl.Message(content="")
    async for chunk in runnable.astream(
        {"question": message.content},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await res.stream_token(chunk)
    
    await res.update()

if __name__ == '__main__':
    print('Hello, Chainlit')