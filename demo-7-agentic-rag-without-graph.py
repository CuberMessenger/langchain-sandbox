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


class DummyToolClass:

    @tool
    @staticmethod
    def TTLUTS_retriever(query: str) -> str:
        """
        A retriever tool for the first 8 chapters of TTLUTS (Twenty Thousand Leagues Under The Sea).

        Args:
            query: keyword or content to search for in the book.
        """

        vector_store = VectorStoreCache.get_vector_store()
        docs = vector_store.similarity_search(query, k=3)
        return "\n\n".join([doc.page_content for doc in docs])

    @tool
    @staticmethod
    def modo_calculator(expression: str) -> str:
        """
        A calculator that compute the "modo" of a given string format expression.

        Args:
            expression: a string.
        """

        modo = f"!!{expression}!!modo!!{expression}!!"
        return modo


def main():
    tool_dictionary = {
        "TTLUTS_retriever": DummyToolClass.TTLUTS_retriever,
        "modo_calculator": DummyToolClass.modo_calculator,
    }

    query_or_respond_prompt = "You are a helpful assistant with some powerful tools. If you think no tools are needed to answer the given question, response nothing."
    query_or_respond_model = get_google_model(
        "gemini-2.5-flash-lite-preview-06-17"
    ).bind_tools(tool_dictionary.values())

    response_prompt = "You are a helpful assistant and every your response ends with an exclamation mark."
    response_model = get_google_model("gemini-2.5-flash")

    # question = (
    #     "What happened on April 13, 1867, in the Twenty Thousand Leagues Under The Sea?"
    # )

    def workflow(question):
        print("#" * 0x7F)

        messages = [
            SystemMessage(content=query_or_respond_prompt),
            HumanMessage(content=question),
        ]

        tool_messages = []
        response = query_or_respond_model.invoke(messages)
        if len(response.tool_calls) == 0:
            print("No tool calls detected. Response:")
        for tool_call in response.tool_calls:
            tool = tool_dictionary[tool_call["name"]]
            tool_message = tool.invoke(tool_call)
            tool_messages.append(tool_message)
            print(f"{tool_call['name']} called!")

        messages = [
            SystemMessage(content=response_prompt),
            HumanMessage(content=question),
            response,  # the tool calling AI message is essential
        ] + tool_messages

        response = response_model.invoke(messages)
        print(response)
        print()

    # workflow(
    #     "What happened on April 13, 1867, in the Twenty Thousand Leagues Under The Sea?"
    # )
    workflow(
        "What happened on April 13, 1867, in the Twenty Thousand Leagues Under The Sea? By the way, what is the modo of the word 'cat'?"
    )
    # workflow("Tell me about Pandora in the Greek myth.")
    # workflow("Tell me about Pandora in the Greek myth. Also, what is the modo of 'Pandora'?")

    # response = query_or_respond_model.invoke(messages)
    # print("#" * 0x7F)
    # print(response)

    # if response.tool_calls is not None:
    #     for tool_call in response.tool_calls:
    #         tool = tool_dict[tool_call["name"].lower()]
    #         tool_message = tool.invoke(tool_call)
    #         messages.append(tool_message)

    #     response = response_model.invoke(messages)
    #     print("#" * 0x7F)
    #     print(response)


if __name__ == "__main__":
    main()
