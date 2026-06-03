import sqlite3

from dotenv import load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, MessagesState, StateGraph

from nodes import run_agent_reasoning, tool_node

load_dotenv()

AGENT_REASON = "agent_reason"
ACT = "act"
LAST = -1


def should_continue(state: MessagesState) -> str:
    if not state["messages"][LAST].tool_calls:
        return END
    return ACT


flow = StateGraph(MessagesState)

flow.add_node(AGENT_REASON, run_agent_reasoning)
flow.set_entry_point(AGENT_REASON)
flow.add_node(ACT, tool_node)

flow.add_conditional_edges(AGENT_REASON, should_continue, {END: END, ACT: ACT})

flow.add_edge(ACT, AGENT_REASON)

# SQLite checkpointer for persistent session storage
sqlite_conn = sqlite3.connect("checkpoints.db", check_same_thread=False)
checkpointer = SqliteSaver(sqlite_conn)
checkpointer.setup()

app = flow.compile(checkpointer=checkpointer)


def list_thread_ids() -> list[str]:
    """Return all thread_ids that have checkpointed sessions."""
    cursor = sqlite_conn.execute(
        "SELECT DISTINCT thread_id FROM checkpoints ORDER BY thread_id"
    )
    return [row[0] for row in cursor.fetchall()]


app.get_graph().draw_mermaid_png(output_file_path="flow.png")

if __name__ == "__main__":

    print("Hello ReAct LangGraph with Function Calling")
    res = app.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "What is the currenttemperature in Tokyo? List it and then triple it",
                }
            ]
        },
        config={"configurable": {"thread_id": "default"}},
    )
    print(res["messages"][-1].content)
