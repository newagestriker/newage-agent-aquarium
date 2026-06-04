from typing import List, TypedDict


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        web_search: whether to add search
        documents: list of documents
        end_app: whether to directly end the app to continue to retrieve the document
    """

    question: str
    generation: str
    websearch: bool
    end_app: bool
    documents: List[str]
    websearch_iteration: int
