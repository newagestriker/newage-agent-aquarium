from dotenv import load_dotenv
from langgraph.graph import END, StateGraph

from .chains.router import router
from .consts import GENERATE, GENERATE_END, GRADE_DOCUMENTS, RETRIEVE, WEBSEARCH
from .nodes.generate import generate
from .nodes.grade_documents import grade_documents
from .nodes.retrieve import retrieve
from .nodes.web_search import web_search
from .state import GraphState
from .chains.answer_grader import answer_grader_chain
from .chains.hallucination_grader import hallucination_grader_chain

load_dotenv()


def route_question(state: GraphState) -> GraphState:
    print("---ROUTE QUESTION---")
    result = router.invoke({"question": state["question"]})
    return GENERATE_END if result.end_app else RETRIEVE


def decide_to_generate(state: GraphState) -> GraphState:
    print("---ASSESS GRADED DOCUMENTS---")
    return WEBSEARCH if state["websearch"] else GENERATE


def generate_end(state: GraphState) -> GraphState:
    return {
        **state,
        "generation": "I can only answer questions related to aquariums and fishkeeping.",
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
            return END
        else:
            print("Answer is not good, web searching.")
            return WEBSEARCH
    else:
        print("Answer is hallucinated, retrying generation.")
        return GENERATE


workflow = StateGraph(GraphState)

workflow.add_node(GENERATE_END, generate_end)
workflow.add_node(GENERATE, generate)
workflow.add_node(RETRIEVE, retrieve)
workflow.add_node(GRADE_DOCUMENTS, grade_documents)
workflow.add_node(WEBSEARCH, web_search)


workflow.set_conditional_entry_point(
    route_question,
    {
        GENERATE_END: GENERATE_END,
        RETRIEVE: RETRIEVE,
    },
)
workflow.add_edge(RETRIEVE, GRADE_DOCUMENTS)
workflow.add_conditional_edges(
    GRADE_DOCUMENTS,
    decide_to_generate,
    {
        WEBSEARCH: WEBSEARCH,
        GENERATE: GENERATE,
    },
)
workflow.add_edge(WEBSEARCH, GENERATE)
workflow.add_edge(GENERATE, END)
workflow.add_conditional_edges(
    GENERATE,
    decide_to_websearch_or_conclude_or_retry,
    {WEBSEARCH: WEBSEARCH, END: END, GENERATE: GENERATE},
)
app = workflow.compile()

app.get_graph().draw_mermaid_png(output_file_path="graph.png")

result = app.invoke({"question": "What is Epistylis?"})

print(result["generation"])