from typing import Any, Dict

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_tavily import TavilySearch

from graph.state import GraphState
from helper.web_to_documents import web_to_documents
from llms.local_net_llm_gemma import llm

load_dotenv()

# Initialize TavilySearch with API key from environment
web_search_tool = TavilySearch(max_results=6)

llm_web_search_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an assistant that modifies a question to find information related to aquariums and fishkeeping. "
            "When given a question, rephrase it to be more specific to aquariums and fishkeeping to retrieve relevant information. "
            "If the question is already specific to aquariums and fishkeeping, keep it as is. "
            "Make sure to include aquarium-related keywords to get the most relevant search results."
            "If web search iteration is greater than 1, make the question more specific to aquariums and fishkeeping and include more relevant keywords to ensure you get useful information from the web search.",
        ),
        ("human", "Question: {question}\n\n Iteration: {websearch_iteration}\n\n Rephrased Question:"),
    ]
)

question_rephraser = llm_web_search_prompt | llm | StrOutputParser()


def web_search(state: GraphState) -> Dict[str, Any]:
    question = state["question"]
    iteration = state.get("websearch_iteration", 0) + 1
    print(f"---PERFORM WEB SEARCH--- {iteration}")
    rephrased_question = question_rephraser.invoke({"question": question, "websearch_iteration": iteration})
    print("Rephrased Question for Web Search:", rephrased_question)
    results = web_search_tool.invoke({"query": rephrased_question})["results"]
    webresults = web_to_documents({"results": results})
    return {"documents": [webresults], "question": question, "websearch_iteration": iteration}
