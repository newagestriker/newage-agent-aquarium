from dotenv import load_dotenv
from langchain_core.tools import tool
#from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

load_dotenv()

@tool
def triple(num: float) -> float:
    """
    Triple a number.
    
    Args:
        num: a number to triple
    
    Returns: 
        the triple of the input number
    """
    return float(num) * 3

tools = [TavilySearch(max_results=1), triple]

#llm = ChatOllama(model="qwen3.5:9b", temperature=0).bind_tools(tools)

llm = ChatOpenAI(
    base_url="http://192.168.1.110:8080/v1",
    api_key="not-needed",
    model="gemma-4-e2b-it",
).bind_tools(tools)
