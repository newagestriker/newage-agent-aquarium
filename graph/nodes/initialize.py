from typing import Any, Dict
from graph.state import GraphState

def initialize(state: GraphState) -> Dict[str, Any]:
    print("---INITIALIZE---")
    return {**state, "websearch_iteration": 0, "messages": state.get("messages",[])}