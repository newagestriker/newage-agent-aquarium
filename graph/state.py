from typing import List, TypedDict
from langchain_core.messages import BaseMessage


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        web_search: whether to add search
        documents: list of documents
        end_app: whether to directly end the app to continue to retrieve the document
        websearch_iteration: number of times we've web searched
        messages: list of messages in the conversation
    """

    question: str
    generation: str
    websearch: bool
    end_app: bool
    documents: List[str]
    websearch_iteration: int
    messages: List[BaseMessage]
