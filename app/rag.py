
import asyncio
import os
import pickle
from dotenv_flow import dotenv_flow
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.documents.base import Document
from langchain_community.document_loaders import GitLoader
from langchain_chroma import Chroma
from langchain_redis import RedisConfig, RedisVectorStore
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai.chat_models import ChatOpenAI


dotenv_flow("dev")


def save_docs(clone_url: str, repo_path: str, doc_path: str) -> None:
    loader = GitLoader(
        clone_url=clone_url,
        repo_path=repo_path,
        branch="master",  # "main"
        file_filter=lambda x: x.endswith(".mdx"),
    )
    raw_docs = loader.load()
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = splitter.split_documents(raw_docs)
    print(len(raw_docs), len(docs))
    with open(doc_path, "rb") as f:
        pickle.dump(docs, f)

def load_docs(doc_path: str) -> list[Document]:
    with open(doc_path, "rb") as f:
        docs = pickle.load(f)
    return docs

async def create_db(docs: list[Document]|None=None) -> Chroma:
    embeddings = OpenAIEmbeddings()
    db = Chroma(
        collection_name="langchain",
        embedding_function=embeddings,
        persist_directory="data/chroma",
    )
    if docs is not None:
        await db.aadd_documents(docs)
    return db

async def create_redis_db(docs: list[Document]|None=None) -> RedisVectorStore:
    """
    Create a RedisVectorStore instance with the given documents.
    https://python.langchain.com/docs/integrations/vectorstores/redis/
    """
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
    redis_config = RedisConfig(
        index_name="langchain",
        redis_url=REDIS_URL,
        metadata_schema=[
            {"name": "source", "type": "tag"},
            {"name": "file_path", "type": "tag"},
            {"name": "file_name", "type": "tag"},
            {"name": "file_type", "type": "tag"}
        ]
    )
    embeddings = OpenAIEmbeddings()
    db = RedisVectorStore(embeddings, redis_config)
    if docs is not None:
        ids = await db.aadd_documents(docs)
        print(f"Added {len(ids)} documents to Redis")
    return db

def create_template() -> ChatPromptTemplate:
    human_prompt = """You are a helpful assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question: {question} 
Context: {context} 
Answer: 
"""
    template = ChatPromptTemplate([("human", human_prompt)])
    return template

def format_docs(docs: list[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)

async def main():
    clone_url = "https://github.com/langchain-ai/langchain"
    repo_path = "langchain"
    docs_path = "data/docs.pkl"
    if os.path.exists(docs_path):
        docs = load_docs(docs_path)
    else:
        save_docs(clone_url, repo_path, docs_path)
        docs = load_docs(docs_path)
    
    query = "AWSのS3からデータを読み込むためのDocument Loaderはありますか? 日本語で答えてください。"

    print(docs[0].metadata)
    ## RAG Retrieval
    #db = await create_db()
    db = await create_redis_db(docs)
    retriever = db.as_retriever()
    context_docs = retriever.invoke(query, k=5)
    first_doc = context_docs[0]
    print(f"metadata: {first_doc.metadata}")
    #print(first_doc.page_content)

    ## Retrieval QA
    llm = ChatOpenAI(model="gpt-4o")
    prompt = create_template()
    qa_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt | llm | StrOutputParser()
    )
    ret = qa_chain.invoke(input=query)
    print(ret)


if __name__ == '__main__':
    asyncio.run(main())