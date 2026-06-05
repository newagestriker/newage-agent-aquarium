from typing import Any, Dict
from langchain_core.messages import AIMessage

from graph.chains.generation import generation_chain
from graph.state import GraphState


def generate(state: GraphState) -> Dict[str, Any]:
    print("---GENERATE---")
    question = state["question"]
    context = state["documents"]

    generation = generation_chain.invoke({"question": question, "context": context})
    # Add the AI response to the messages list
    messages = state.get("messages", [])
    messages.append(AIMessage(content=generation))
    return {**state, "generation": generation, "messages": messages}
