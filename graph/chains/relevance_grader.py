from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from pydantic import BaseModel, Field

from llms.local_net_llm_gemma import llm


class RelevanceGrade(BaseModel):
    """Binary score for relevance of documents retrieved."""

    grade: bool = Field(
        description="Whether the retrieved document is relevant to the question. Only answer true if you are highly confident (>0.8). Answer with false for borderline or irrelevant documents.",
    )


structured_llm_grader = llm.with_structured_output(RelevanceGrade)

system_message = """You are a strict grader assessing relevance of a retrieved document to a user question.

MARK AS RELEVANT ONLY IF:
- The document directly answers or helps answer the specific question
- It contains specific facts, data, or examples directly applicable to the question
- It provides necessary context that is essential to understanding the answer
- The connection to the question is clear and non-obvious

MARK AS NOT RELEVANT IF:
- The document only shares surface-level keywords with the question
- It's tangentially related but doesn't actually help answer the question
- It's generic background information that applies to many topics
- It mentions the topic but doesn't provide useful information for this specific question
- The relevance requires significant inference or logical leaps

Be STRICT. Only mark documents as relevant when you are confident they genuinely help answer the user's question."""

grader_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_message),
        (
            "user",
            "Question: \n\n {question} \n\n Retrieved Document: \n\n {documents} \n\n Is the document truly relevant and helpful for answering this question? Only answer 'true' if you are highly confident (>80% certain). Otherwise answer 'false'.",
        ),
    ]
)

relevance_grader: RunnableSequence = grader_prompt | structured_llm_grader
