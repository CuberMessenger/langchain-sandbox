from utility import *

from langchain_core.tools import tool
from langchain_core.tools import Tool
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_core.vectorstores import InMemoryVectorStore

from typing import Literal
from pydantic import BaseModel, Field


class VectorStoreCache:
    VECTOR_STORE = None

    @staticmethod
    def load_vector_store():
        embedding_model = get_google_embedding_model()

        VectorStoreCache.VECTOR_STORE = InMemoryVectorStore.load(
            path="TTLUTS-First-8-Chapters-vector-store.json",
            embedding=embedding_model,
        )
        monkey_patch_in_memory_vector_store(VectorStoreCache.VECTOR_STORE)

    @staticmethod
    def get_vector_store():
        if VectorStoreCache.VECTOR_STORE is None:
            VectorStoreCache.load_vector_store()
        return VectorStoreCache.VECTOR_STORE


@tool
def TTLUTS_retriever(query: str) -> str:
    """A retriever tool for the first 8 chapters of TTLUTS (Twenty Thousand Leagues Under The Sea).

    Args:
        query: keyword or content to search for in the book.
    """

    vector_store = VectorStoreCache.get_vector_store()
    docs = vector_store.similarity_search(query, k=3)
    return "\n\n".join([doc.page_content for doc in docs])


def main():
    query_or_response_model = get_google_model().bind_tools([TTLUTS_retriever])
    response_model = get_google_model()

    messages = [
        SystemMessage(
            "You are a helpful assistant and every your response ends with an exclamation mark."
        ),
        HumanMessage(
            "What happened on April 13, 1867, in the Twenty Thousand Leagues Under The Sea?"
        ),
        # HumanMessage(
        #     "Tell me about the myth figure Pandora and her box."
        # ),
    ]

    tool_dict = {
        "ttluts_retriever": TTLUTS_retriever,
    }

    response = query_or_response_model.invoke(messages)
    print("#" * 0x7F)
    print(response)

    if response.tool_calls is not None:
        for tool_call in response.tool_calls:
            tool = tool_dict[tool_call["name"].lower()]
            tool_message = tool.invoke(tool_call)
            messages.append(tool_message)

        response = response_model.invoke(messages)
        print("#" * 0x7F)
        print(response)


if __name__ == "__main__":
    main()
