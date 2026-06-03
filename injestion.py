import os

from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

with open("urls.txt") as f:
    urls = f.read().splitlines()

docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250, chunk_overlap=0
)
doc_splits = text_splitter.split_documents(docs_list)

QdrantVectorStore.from_documents(
    documents=doc_splits,
    embedding=OllamaEmbeddings(model="nomic-embed-text"),
    collection_name=os.getenv("QDRANT_COLLECTION"),
    url=os.getenv("QDRANT_URL"),
)

