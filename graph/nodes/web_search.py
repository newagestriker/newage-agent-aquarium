from typing import Any, Dict
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_tavily import TavilySearch

from graph.state import GraphState

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
            "Make sure to include aquarium-related keywords to get the most relevant search results.",
        ),
        ("human", "Question: {question}"),
    ]
)

question_rephraser = llm_web_search_prompt | llm | StrOutputParser()


def web_search(state: GraphState) -> Dict[str, Any]:
    print("---PERFORM WEB SEARCH---")
    question = state["question"]
    rephrased_question = question_rephraser.invoke({"question": question})
    results = web_search_tool.invoke({"query": rephrased_question})["results"]
    formatted_results = "\n".join(result["content"] for result in results)

    webresults = Document(page_content=formatted_results)
    return {"documents": [webresults], "question": question}
