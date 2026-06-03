from typing import Any, Dict

from graph.chains.relevance_grader import relevance_grader
from graph.state import GraphState


def grade_documents(state: GraphState) -> Dict[str, Any]:
    print("---GRADE DOCUMENTS---")
    result = relevance_grader.invoke(
        {"question": state["question"], "documents": state["documents"]}
    )
    return {**state, "websearch": not result.grade}
