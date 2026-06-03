from typing import Any, Dict

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_tavily import TavilySearch

from graph.state import GraphState

load_dotenv()

# Initialize TavilySearch with API key from environment
web_search_tool = TavilySearch(max_results=3)


def web_search(state: GraphState) -> Dict[str, Any]:
    print("---PERFORM WEB SEARCH---")
    question = state["question"]
    results = web_search_tool.invoke({"query": question})["results"]
    formatted_results = "\n".join(result["content"] for result in results)

    webresults = Document(page_content=formatted_results)
    return {"document": [webresults], "question": question}
