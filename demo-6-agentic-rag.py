from utility import *

from langchain_core.tools import tool
from langgraph.graph import MessagesState
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition

from IPython.display import Image, display


def node1():
    pass


def node2():
    pass


@tool
def useful_tool1():
    """A useful tool that does something."""
    pass


@tool
def useful_tool2():
    """Another useful tool that does something else."""
    pass


def node3():
    pass


def main():
    workflow = StateGraph(MessagesState)

    workflow.add_node(node1)
    workflow.add_node(node2)
    workflow.add_node(node3)
    workflow.add_node("useful_tools", ToolNode([useful_tool1, useful_tool2]))

    workflow.add_edge(START, "node1")
    workflow.add_edge("node1", "node2")

    workflow.add_conditional_edges(
        source="node2",
        path=tools_condition, # it outputs "tools" or END
        path_map={"tools": "useful_tools", END: "node3"},
    )

    workflow.add_edge("useful_tools", "node3")

    workflow.add_edge("node3", END)

    graph = workflow.compile()
    print("Graph compiled successfully!")

    image = Image(graph.get_graph().draw_mermaid_png())
    # save the image to a file
    with open("workflow_graph.png", "wb") as f:
        f.write(image.data)


if __name__ == "__main__":
    main()
