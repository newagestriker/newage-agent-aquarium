from typing import Any
from langchain_core.documents import Document
def web_to_documents(web_search_results: Any) -> Document:
    """
    Convert web search results into a Document format.

    Args:
        web_search_results: The raw results from the web search tool.

    Returns:
        A Document object containing the formatted web search results.
    """
    formatted_results = "\n".join(result["content"] for result in web_search_results["results"])
    return Document(page_content=formatted_results)
