from langmem.short_term import SummarizationNode
from llms.local_net_llm_gemma import llm

summarization_node = SummarizationNode(  
    token_counter=llm.get_num_tokens_from_messages,
    model=llm,
    max_tokens=256,
    max_tokens_before_summary=256,
    max_summary_tokens=128,
)
