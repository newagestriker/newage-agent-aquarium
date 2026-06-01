
from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph
from .state import GraphState
from .nodes.retrieve import retrieve
from .consts import RETRIEVE, GRADE_DOCUMENTS
from .chains.relevance_grader import relevance_grader

load_dotenv()


workflow = StateGraph(GraphState)

workflow.add_node(RETRIEVE, retrieve)
workflow.add_node(GRADE_DOCUMENTS, relevance_grader)

workflow.add_edge(START, RETRIEVE)
workflow.add_edge(RETRIEVE, GRADE_DOCUMENTS)
workflow.add_edge(GRADE_DOCUMENTS, END)

app = workflow.compile()

app.get_graph().draw_mermaid_png(output_file_path="graph.png")

app.invoke({"question": "How should I treat ich?"})