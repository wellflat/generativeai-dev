
import asyncio
import os
import pickle
from dotenv_flow import dotenv_flow
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.documents.base import Document
from langchain_community.document_loaders import GitLoader
from langchain_chroma import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings

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


async def main():
    clone_url = "https://github.com/langchain-ai/langchain"
    repo_path = "langchain"
    docs_path = "data/docs.pkl"
    if os.path.exists(docs_path):
        docs = load_docs(docs_path)
    else:
        save_docs(clone_url, repo_path, docs_path)
        docs = load_docs(docs_path)
    
    query = "AWSのS3からデータを読み込むためのDocument Loaderはありますか？"
    db = await create_db()
    retriever = db.as_retriever()
    context_docs = retriever.invoke(query, k=5)
    first_doc = context_docs[0]
    print(f"metadata: {first_doc.metadata}")
    print(first_doc.page_content)


if __name__ == '__main__':
    asyncio.run(main())