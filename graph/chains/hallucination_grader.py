from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from pydantic import BaseModel, Field

from llms.local_net_llm_gemma import llm


class HallucinationGrade(BaseModel):
    grade: bool = Field(
        description="""Whether the answer contains hallucinated information that is not supported by the retrieved documents.
        Only answer true if you are highly confident (>0.8) that the answer is accurate and grounded in the retrieved documents.
        Answer false if there are any signs of hallucination, fabrication, or unsupported claims.""",
    )


system_message = """You are a strict grader assessing whether an answer contains hallucinated information that is not supported by the retrieved documents.
MARK AS NOT HALLUCINATED (TRUE) ONLY IF:
- The answer is directly supported by specific facts, data, or examples in the retrieved documents
- The answer provides necessary context that is essential to understanding the question and is clearly grounded in the retrieved information (>0.8)
- The answer does not include any information that cannot be traced back to the retrieved documents"""

grader_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_message),
        (
            "user",
            "Question: \n\n {question} \n\n Retrieved Documents: \n\n {documents} \n\n Generated Answer: \n\n {generation} \n\n Is the generated answer free of hallucinated information and fully supported by the retrieved documents? Only answer 'true' if you are highly confident (>80% certain) that the answer is accurate and grounded in the retrieved documents. Otherwise answer 'false'.",
        ),
    ]
)

hallucination_grader_chain: RunnableSequence = (
    grader_prompt | llm.with_structured_output(HallucinationGrade)
)
