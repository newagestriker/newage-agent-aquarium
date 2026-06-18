from langmem.short_term import SummarizationNode
from llms.local_net_llm_gemma import llm
from langchain_core.messages.utils import count_tokens_approximately

summarization_node = SummarizationNode(  
    token_counter=count_tokens_approximately,
    model=llm,
    max_tokens=256,
    max_tokens_before_summary=256,
    max_summary_tokens=128,
)
