from dotenv import load_dotenv
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode

from react import llm, tools

load_dotenv()

SYSYEM_MESSAGE = """
You are a helpful assistant that can use tools to answer questions related to the aquarium.
Do not answer questions that are not related to the aquarium. If a question is not related to the aquarium, respond with "I'm sorry, I can only answer questions related to the aquarium."
You should not make up any information and explicitly inform the user that you cannot answer a question that is not related to the aquarium.
"""


def run_agent_reasoning(state: MessagesState) -> MessagesState:
    """
    Run the agent reasoning node.
    """
    response = llm.invoke(
        [{"role": "system", "content": SYSYEM_MESSAGE}, *state["messages"]]
    )
    # Append the response to existing messages instead of replacing them
    return {"messages": state["messages"] + [response]}


tool_node = ToolNode(tools)
