from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from llms.local_net_llm_gemma import llm


class RouterDecision(BaseModel):
    end_app: bool = Field(
        description="Set to True if the question is NOT related to aquariums or fishkeeping (end the app). Set to False if it IS related (continue to retrieve documents).",
    )


structured_llm_router = llm.with_structured_output(RouterDecision)
router_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a strict aquarium and fishkeeping assistant. "
            "If the user question is related to aquariums, fish, fishkeeping, water quality, tank setup, fish diseases, or aquatic life, set end_app to False. "
            "If the question is about anything else, set end_app to True.",
        ),
        ("user", "User question: {question}"),
    ]
)

router = router_prompt | structured_llm_router
