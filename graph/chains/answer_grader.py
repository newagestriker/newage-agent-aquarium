from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from pydantic import BaseModel, Field

from llms.local_net_llm_gemma import llm


class AnswerGrade(BaseModel):
    """Binary score for accuracy and relevance of an answer to a question."""

    grade: bool = Field(
        description="Whether the answer is accurate and relevant to the question. Only answer true if you are highly confident (>0.8). Answer with false for borderline or irrelevant answers.",
    )


system_message = (
    """You are a strict grader assessing the quality of an answer to a user question.
    MARK AS GOOD ONLY IF: The answer is accurate, relevant, and directly addresses the question with high confidence (>0.8).
    MARK AS BAD IF: 
    - The answer is inaccurate, irrelevant, or only tangentially related to the question, or if you are not highly confident in its quality.
    - The answer does not fully address the question due to missing key information, even if it is partially correct.
    Be STRICT. Only mark answers as good when you are confident they genuinely answer the user's question"""
)

grader_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_message),
        (
            "user",
            "Question: {question}\n\nAnswer: {answer}\n\nIs the answer accurate and relevant to the question? Only answer 'true' if you are highly confident (>0.8). Otherwise answer 'false'.",
        ),
    ]
)

structured_llm_answer_grader = llm.with_structured_output(AnswerGrade)


answer_grader_chain: RunnableSequence = grader_prompt | structured_llm_answer_grader
