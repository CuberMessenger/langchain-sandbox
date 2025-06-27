from utility import *

from langchain_core.tools import tool
from langchain_core.tools import Tool
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_core.vectorstores import InMemoryVectorStore
from langgraph.graph import MessagesState
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition

from typing import Literal
from pydantic import BaseModel, Field
from IPython.display import Image, display


def get_vector_store():
    embedding_model = get_google_embedding_model()

    vector_store = InMemoryVectorStore.load(
        path="TTLUTS-First-8-Chapters-vector-store.json",
        embedding=embedding_model,
    )
    monkey_patch_in_memory_vector_store(vector_store)

    return vector_store


def get_retriever_tool(
    vector_store,
    k=2,
    name="retriever",
    description="Retrieve relevant documents based on a query.",
):
    def retriever(query: str) -> str:
        docs = vector_store.similarity_search(query, k=k)
        return "\n\n".join([doc.page_content for doc in docs])

    retriever_tool = Tool.from_function(
        func=retriever,
        name=name,
        description=description,
    )

    return retriever_tool


# prepare retriever tool
vector_store = get_vector_store()
retrieve_tool = get_retriever_tool(
    vector_store=vector_store,
    k=3,
    name="TTLUTS Retriever",
    description="A retriever tool for the first 8 chapters of TTLUTS (Twenty Thousand Leagues Under The Sea).",
)

query_or_response_model = get_google_model().bind_tools([retrieve_tool])
evaluate_model = get_google_model()
rewrite_model = get_google_model()


def query_or_response_node(state: MessagesState):
    response = query_or_response_model.invoke(state["messages"])
    return {"messages": [response]}


class DocumentScore(BaseModel):
    """A binary score of a document showing its relevance to a user question."""

    binary_score: Literal["yes", "no"] = Field(
        description="Relevance score: 'yes' if relevant, or 'no' if not relevant"
    )


def evaluate_document(
    state: MessagesState,
) -> Literal["response_node", "rewrite_query_node"]:
    """Evaluate the retrieved documents."""

    GRADE_PROMPT = (
        "You are a grader assessing relevance of a retrieved document to a user question. \n "
        "Here is the retrieved document: \n\n {document} \n\n"
        "Here is the user question: {question} \n"
        "If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n"
        "Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."
    )

    question = state["messages"][0].content
    document = state["messages"][-1].content

    prompt = GRADE_PROMPT.format(
        document=document,
        question=question,
    )

    score = (
        evaluate_model.with_structured_output(DocumentScore).invoke(
            [HumanMessage(prompt)]
        )
    ).binary_score

    if score == "yes":
        return "response_node"
    elif score == "no":
        return "rewrite_query_node"
    else:
        raise ValueError(f"Unexpected score: {score}. Expected 'yes' or 'no'.")


def rewrite_query_node(state: MessagesState):
    REWRITE_PROMPT = (
        "Look at the input and try to reason about the underlying semantic intent / meaning.\n"
        "Here is the initial question:"
        "\n ------- \n"
        "{question}"
        "\n ------- \n"
        "Formulate an improved question:"
    )

    question = state["messages"][0].content
    prompt = REWRITE_PROMPT.format(question=question)
    response = rewrite_model.invoke([HumanMessage(prompt)])

    return {"messages": [HumanMessage(response)]}


def response_node(state: MessagesState):
    RESPONSE_PROMPT = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved document to answer the question. "
        "If you don't know the answer, just say that you don't know. "
        "Use three sentences maximum and keep the answer concise.\n"
        "Question: {question} \n"
        "Documents: {document}"
    )

    question = state["messages"][0].content
    document = state["messages"][-1].content
    prompt = RESPONSE_PROMPT.format(question=question, document=document)
    response = query_or_response_model.invoke([HumanMessage(prompt)])
    return {"messages": [response]}


#####################################################


def get_rag_graph(retrieve_tool):
    workflow = StateGraph(MessagesState)

    workflow.add_node(query_or_response_node)
    workflow.add_node("retrive_tool_node", ToolNode([retrieve_tool]))
    workflow.add_node(rewrite_query_node)
    workflow.add_node(response_node)

    workflow.add_edge(START, "query_or_response_node")

    workflow.add_conditional_edges(
        source="query_or_response_node",
        path=tools_condition,  # it outputs "tools" or END
        path_map={"tools": "retrive_tool_node", END: END},
    )

    workflow.add_conditional_edges(
        source="retrive_tool_node",
        path=evaluate_document,  # it outputs "response_node" or "rewrite_query_node"
    )

    workflow.add_edge("response_node", END)
    workflow.add_edge("rewrite_query_node", "query_or_response_node")

    graph = workflow.compile()

    print("Graph compiled successfully!")

    image = Image(graph.get_graph().draw_mermaid_png())
    # save the image to a file
    with open("workflow_graph.png", "wb") as f:
        f.write(image.data)

    return graph


def main():
    # prepare workflow
    rag_graph = get_rag_graph(retrieve_tool)

    # prepare model
    state = MessagesState()

    state["messages"] = [
        SystemMessage(
            "You are a helpful assistant and every your response ends with an exclamation mark."
        ),
        HumanMessage(
            "What happened on April 13, 1867, in the Twenty Thousand Leagues Under The Sea?"
        ),
    ]

    # run the workflow
    print("#" * 0x7F)
    print("Running the RAG workflow...")
    for chunk in rag_graph.stream(state):
        for node, update in chunk.items():
            print("Update from node", node)
            update["messages"][-1].pretty_print()
            print("\n\n")


if __name__ == "__main__":
    main()
