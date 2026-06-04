import os

from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

load_dotenv()

def ingest_data(docs_list: list[Document]) -> None:

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=250, chunk_overlap=0
    )
    doc_splits = text_splitter.split_documents(docs_list)

    QdrantVectorStore.from_documents(
        documents=doc_splits,
        embedding=OllamaEmbeddings(
            model=os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text"), 
            base_url=os.getenv("OLLAMA_BASE_URL", "http://192.168.1.110:11434")
        ),
        collection_name=os.getenv("QDRANT_COLLECTION"),
        url=os.getenv("QDRANT_URL"),
    )

if __name__ == "__main__":
    with open("urls.txt") as f:
        urls = f.read().splitlines()
    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]
    ingest_data(docs_list)
