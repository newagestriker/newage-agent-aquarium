

from typing import Dict

from langchain_protocol import Any

from graph.state import GraphState
from injestion import ingest_data


def injest(state: GraphState) -> Dict[str, Any]:
    print("---INGEST---")
    documents = state["documents"]
    ingest_data(documents)
    return {**state}