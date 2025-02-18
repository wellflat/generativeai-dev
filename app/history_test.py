import os

from dotenv_flow import dotenv_flow
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from langchain_redis import RedisChatMessageHistory

dotenv_flow("dev")

store: dict[str, ChatMessageHistory] = {}

def create_history(session_id: str, redis_url: str|None=None) -> BaseChatMessageHistory:
    if redis_url is None:
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
    history = RedisChatMessageHistory(
        session_id=session_id,  # note: セッションIDにハイフン"-"は使えない
        redis_url=redis_url,
        ttl=600,
    )
    return history

def create_inmemory_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

def create_runnable() -> RunnableWithMessageHistory:
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful AI assistant."),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ]
    )
    llm = ChatOpenAI(model="gpt-4o")
    runnable: Runnable = prompt | llm | StrOutputParser()
    chat_with_history = RunnableWithMessageHistory(
        runnable, create_inmemory_history, input_messages_key="input", history_messages_key="history"
    )
    return chat_with_history


if __name__ == "__main__":
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
    print(f"Connecting to Redis at: {REDIS_URL}")
    runnable = create_runnable()
    session_id = "test_session"
    config: RunnableConfig = {"configurable": {"session_id": session_id}}
    response1 = runnable.invoke({"input": "こんにちは、私は3歳です"}, config=config)
    print("AI Response 1:", response1)

    response2 = runnable.invoke({"input": "私は何歳か覚えていますか?"}, config=config)
    print("AI Response 2:", response2)

    print(runnable.get_session_history(session_id=session_id))
