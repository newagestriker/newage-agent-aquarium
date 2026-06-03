from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    base_url="http://192.168.1.171:11434/v1",
    api_key="not-needed",
    model="qwen3.6:35b-a3b-q4_K_M",
    temperature=0,)