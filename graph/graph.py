import sqlite3
import uuid

from dotenv import load_dotenv
from langgraph.graph import END, StateGraph
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import AIMessage, HumanMessage

from .chains.router import router
from .consts import GENERATE, GENERATE_END, GRADE_DOCUMENTS, RETRIEVE, WEBSEARCH, INJEST, INITIALIZE, SUMMARIZE
from .nodes.generate import generate
from .nodes.grade_documents import grade_documents
from .nodes.retrieve import retrieve
from .nodes.web_search import web_search
from .nodes.injest import injest
from .nodes.initialize import initialize
from .nodes.summarize import summarization_node
from .state import GraphState
from .chains.answer_grader import answer_grader_chain
from .chains.hallucination_grader import hallucination_grader_chain

load_dotenv()


def route_question(state: GraphState) -> GraphState:
    print("---ROUTE QUESTION---")
    question = state["question"]
    query_id = state.get("query_id", "")
    summarized_messages = state.get("summarized_messages",[])
    result = router.invoke({"question": question, "summarized_messages": summarized_messages})
    return GENERATE_END if result.end_app else RETRIEVE


def decide_to_generate(state: GraphState) -> GraphState:
    print("---ASSESS GRADED DOCUMENTS---")
    return WEBSEARCH if state["websearch"] else GENERATE


def generate_end(state: GraphState) -> GraphState:
    generation = "I can only answer questions related to aquariums and fishkeeping."
    return {
        **state,
        "generation": generation,
        "messages": [AIMessage(content=generation, id=str(uuid.uuid4()))],
    }


def decide_to_websearch_or_conclude_or_retry(state: GraphState) -> GraphState:
    print("---DECIDE TO WEBSEARCH OR CONCLUDE OR RETRY ---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]
    
    hallucination_result = hallucination_grader_chain.invoke(
        {
            "question": question,
            "documents": documents,
            "generation": generation,
        }
    )
    answer_result = answer_grader_chain.invoke(
        {"question": question, "answer": generation}
    )
    if hallucination_result.grade:
        if answer_result.grade:
            print("Answer is good, concluding.")
            return SUMMARIZE
        else:
            print("Answer is not good, web searching.")
            return WEBSEARCH
    else:
        print("Answer is hallucinated, retrying generation.")
        return GENERATE

def decide_to_injest_if_web_generated(state: GraphState) -> GraphState:
    print("---DECIDE TO INJEST IF WEB GENERATED---")
    if state["websearch"] and state["documents"]:
        return INJEST
    else:
        return END
workflow = StateGraph(GraphState)

workflow.add_node(GENERATE_END, generate_end)
workflow.add_node(GENERATE, generate)
workflow.add_node(RETRIEVE, retrieve)
workflow.add_node(GRADE_DOCUMENTS, grade_documents)
workflow.add_node(WEBSEARCH, web_search)
workflow.add_node(INJEST, injest)
workflow.add_node(INITIALIZE, initialize)
workflow.add_node(SUMMARIZE, summarization_node)
workflow.set_conditional_entry_point(
    route_question,
    {
        GENERATE_END: GENERATE_END,
        RETRIEVE: RETRIEVE,
    },
)
workflow.add_edge(RETRIEVE, INITIALIZE)
workflow.add_edge(INITIALIZE, GRADE_DOCUMENTS)
workflow.add_conditional_edges(
    GRADE_DOCUMENTS,
    decide_to_generate,
    {
        WEBSEARCH: WEBSEARCH,
        GENERATE: GENERATE,
    },
)
workflow.add_edge(WEBSEARCH, GENERATE)
workflow.add_conditional_edges(
    GENERATE,
    decide_to_websearch_or_conclude_or_retry,
    {WEBSEARCH: WEBSEARCH, SUMMARIZE: SUMMARIZE, GENERATE: GENERATE},
)

workflow.add_conditional_edges(
    SUMMARIZE,
    decide_to_injest_if_web_generated,
    {INJEST: INJEST, END: END},
)

def setup_checkpointer():
    try:
        sqlite_conn = sqlite3.connect("checkpoints.db", check_same_thread=False)
        checkpointer = SqliteSaver(sqlite_conn)
        checkpointer.setup()
        return sqlite_conn, checkpointer
    except Exception as e:
        # If we can't connect to the database, create a temporary in-memory one for testing
        print(f"Warning: Could not connect to database: {e}")
        sqlite_conn = sqlite3.connect(":memory:", check_same_thread=False)
        checkpointer = SqliteSaver(sqlite_conn)
        checkpointer.setup()
        return sqlite_conn, checkpointer

sqlite_conn, checkpointer = setup_checkpointer()

def list_thread_ids() -> list[str]:
    """Return all thread_ids that have checkpointed sessions."""
    try:
        cursor = sqlite_conn.execute(
            "SELECT DISTINCT thread_id FROM checkpoints ORDER BY thread_id"
        )
        return [row[0] for row in cursor.fetchall()]
    except Exception:
        # If there's an issue with the database, return empty list
        return []


app = workflow.compile(checkpointer=checkpointer)

# Ensure the checkpointer is properly set up for state persistence
# The checkpointer will automatically handle state management for each thread_id
# This allows conversation context to be maintained across multiple invocations

if __name__ == '__main__':
    app.get_graph().draw_mermaid_png(output_file_path="graph.png")

    result = app.invoke({"question": "What is a Walstad Tank?", "query_id": str(uuid.uuid4())},  config={"configurable": {"thread_id": "session-1"}})
    print(result["generation"])
    result = app.invoke({"question": "How to set it up?", "query_id": str(uuid.uuid4())},  config={"configurable": {"thread_id": "session-1"}})

    print(result["generation"])
    print(result["messages"])
    print(result["summarized_messages"])