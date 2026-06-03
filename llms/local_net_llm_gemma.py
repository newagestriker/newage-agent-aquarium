from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    base_url="http://192.168.1.110:8080/v1",
    api_key="not-needed",
    model="gemma-4-e2b-it",
    temperature=0,
)
