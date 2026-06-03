from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from main import app as graph_app
from main import list_thread_ids

app = FastAPI(title="Newage Aquarium Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str


class QueryRequest(BaseModel):
    query: str
    thread_id: str  # Session identifier managed by caller


class QueryResponse(BaseModel):
    response: str


class MessageItem(BaseModel):
    role: str
    content: str


@app.post("/api/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Stateful endpoint — LangGraph checkpointer manages conversation history.
    Caller only needs to send the new message and a thread_id (session ID).
    """
    try:
        messages = [HumanMessage(content=request.query)]

        # Invoke the graph with checkpointer managing session state
        result = graph_app.invoke(
            {"messages": messages},
            config={"configurable": {"thread_id": request.thread_id}},
        )

        last_message = result["messages"][-1]
        return {"response": last_message.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/history/{thread_id}")
async def get_history(thread_id: str):
    """
    Fetch conversation history for a given thread_id.
    Returns all messages in the conversation.
    """
    try:
        config = {"configurable": {"thread_id": thread_id}}
        state = graph_app.get_state(config)

        messages = []
        for msg in state.values["messages"]:
            role = "user" if msg.type == "human" else "assistant"
            messages.append({"role": role, "content": msg.content})

        return {"thread_id": thread_id, "messages": messages}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/threads")
async def threads():
    """
    List all thread_ids (sessions) that have checkpointed state.
    """
    try:
        thread_ids = list_thread_ids()
        return {"thread_ids": thread_ids}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/threads")
async def create_thread():
    """
    Create a new thread_id for a fresh conversation session.
    """
    import uuid

    thread_id = str(uuid.uuid4())
    return {"thread_id": thread_id}


@app.get("/api/health")
async def health():
    """
    Health check endpoint.
    """
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
