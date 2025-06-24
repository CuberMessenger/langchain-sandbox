from utility import *
from langchain_core.tools import Tool
from langchain_core.vectorstores import InMemoryVectorStore
from langchain.tools.retriever import create_retriever_tool

def get_retriever_tool(vector_store, k=2, name="retriever", description="Retrieve relevant documents based on a query."):
    def retriever(query: str) -> str:
        docs = vector_store.similarity_search(query, k=k)
        return "\n\n".join([doc.page_content for doc in docs])
    
    retriever_tool = Tool.from_function(
        func=retriever,
        name=name,
        description=description,
    )

    return retriever_tool


def main():
    embedding_model = get_google_embedding_model()

    vector_store = InMemoryVectorStore.load(
        path="TTLUTS-First-8-Chapters-vector-store.json",
        embedding=embedding_model,
    )
    monkey_patch_in_memory_vector_store(vector_store)

    # docs = vector_store.similarity_search("April 13, 1867", k=1)

    # for doc in docs:
    #     print("#" * 0x7F)
    #     print(doc)
    #     print()


    retriever_tool = get_retriever_tool(
        vector_store=vector_store,
        k=1,
        name="TTLUTS Retriever",
        description="A retriever tool for the first 8 chapters of TTLUTS (Twenty Thousand Leagues Under The Sea).",
    )

    print("#" * 0x7F)

    docs = retriever_tool.invoke({
        "query": "1867",
    })

    print("#" * 0x7F)
    print(docs)



if __name__ == "__main__":
    main()
