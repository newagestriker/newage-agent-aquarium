from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from pydantic import BaseModel, Field

from llms.local_net_llm_gemma import llm

class AnswerGrade(BaseModel):
    """Binary score for accuracy and relevance of an answer to a question."""

    grade: bool = Field(
        description="Whether the answer is accurate and relevant to the question. Only answer true if you are highly confident (>0.8). Answer with false for borderline or irrelevant answers.",
    )

system_message = """You are a strict grader assessing the quality of an answer to a user question."""

grader_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_message),
        ("user", "Question: {question}\n\nAnswer: {answer}\n\nIs the answer accurate and relevant to the question? Only answer 'true' if you are highly confident (>0.8). Otherwise answer 'false'."),
    ]
)

structured_llm_answer_grader = llm.with_structured_output(AnswerGrade)


answer_grader_chain: RunnableSequence = grader_prompt | structured_llm_answer_grader