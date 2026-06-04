import os
from typing import Any, Dict

from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from ..state import GraphState

# Cache embeddings to avoid recreating on every call
_embeddings_cache = None


def _get_embeddings():
    """Get or create embeddings instance (cached)."""
    global _embeddings_cache
    if _embeddings_cache is None:
        _embeddings_cache = OllamaEmbeddings(model="nomic-embed-text")
    return _embeddings_cache


def retrieve(state: GraphState) -> Dict[str, Any]:
    """Retrieves relevant documents from the vector store based on the user's question.

    Args:
        state: GraphState containing the user's question

    Returns:
        Dictionary with 'documents' list and 'question' string
    """
    print("---RETRIEVE---")
    question = state["question"]
    print(f"Question: {question}")

    try:
        client = QdrantClient(
            host=os.getenv("QDRANT_HOST"), port=int(os.getenv("QDRANT_PORT", 6333))
        )

        embeddings = _get_embeddings()

        vector_store = QdrantVectorStore(
            client=client,
            collection_name=os.getenv("QDRANT_COLLECTION"),
            embedding=embeddings,
        )

        # Use similarity search with explicit parameters
        # k=4 retrieves top 4 documents (increased from default)
        # score_threshold can filter low-relevance results
        documents = vector_store.similarity_search(query=question, k=6)

        print(f"Retrieved {len(documents)} documents")
        for i, doc in enumerate(documents, 1):
            print(f"  Doc {i}: {doc.metadata.get('title', 'Unknown')}")

    except Exception as e:
        print(f"ERROR during retrieval: {type(e).__name__}: {e}")
        documents = []

    return {"documents": documents, "question": question}
