
import os
from typing import Dict, Any

from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from ..state import GraphState


def retrieve(state: GraphState) -> Dict[str, Any]:
    """Retrieves relevant documents from the vector store based on the user's question."""
    print("---RETRIEVE---")
    question = state["question"]

    client = QdrantClient(
    host=os.getenv("QDRANT_HOST"),
    port=os.getenv("QDRANT_PORT")
    )
    retriever = QdrantVectorStore(
    client=client,
    collection_name=os.getenv("QDRANT_COLLECTION"),
    embedding=OllamaEmbeddings(model="nomic-embed-text"),
    ).as_retriever()

    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}