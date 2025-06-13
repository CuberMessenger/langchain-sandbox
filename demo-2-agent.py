from utility import *

from langchain_core.tools import Tool
from langchain_google_community import GoogleSearchAPIWrapper
from langchain_core.messages import HumanMessage, SystemMessage

from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent


def main():
    memory = MemorySaver()
    model = get_google_model(name="gemini-2.0-flash")

    get_google_cse_id()
    google_search_wrapper = GoogleSearchAPIWrapper(k=1)
    search_tool = Tool(
        name = "google_search",
        description="Search Google for recent results.",
        func=google_search_wrapper.run,
    )
    tools = [search_tool]

    agent = create_react_agent(model, tools, checkpointer=memory)

    config = {
        "configurable": {
            "thread_id": "fine-blackberry",
        }
    }

    messages = [
        SystemMessage("You are a helpful assistant and every your response ends with an exclamation mark."),
        HumanMessage(
            "Hi, I'm Frink and I live in Seoul. Please search some news happen around me.",
        ),
    ]

    for step in agent.stream(
        {"messages": messages},
        config=config,
        stream_mode="values"
    ):
        step["messages"][-1].pretty_print()


if __name__ == "__main__":
    main()
