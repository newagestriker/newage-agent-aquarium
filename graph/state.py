from typing import List, TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langmem.short_term import RunningSummary
from langgraph.graph.message import add_messages


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
        query_id: Id of the query from the client
        summarized_messages: Summary of the past coversation
    """

    question: str
    generation: str
    websearch: bool
    end_app: bool
    documents: List[str]
    websearch_iteration: int
    messages: Annotated[List[BaseMessage], add_messages]
    query_id: str
    summarized_messages: List[BaseMessage]
