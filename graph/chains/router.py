from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from llms.local_net_llm_gemma import llm

class RouterQuery(BaseModel):
    database: Literal["vectorstore", "websearch"] = Field(
        ...,
        description="Given a user question choose to route it to web search or a vectorstore.",
    )

structured_llm_router = llm.with_structured_output(RouterQuery)
router_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant that routes user questions to the appropriate tool."),
        (
            "user",
            "Given a user question, choose to route it to web search or a vectorstore. Always choose one of the two options and never refuse to answer. Answer in the following format: {database: 'vectorstore'} or {database: 'websearch'}. User question: {question}",
        ),
    ]
)

