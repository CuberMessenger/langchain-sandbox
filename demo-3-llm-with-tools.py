from utility import *

from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage


@tool
def modo(a: int, b: int) -> int:
    """Compute the modo of two integers.

    Args:
        a (int): The first integer.
        b (int): The second integer.
    """

    return a * a * b + b * b * a


@tool
def multiply(a: int, b: int) -> int:
    """Multiply a and b.

    Args:
        a: first int
        b: second int
    """
    return a * b


@tool
def add(a: int, b: int) -> int:
    """Add a and b.

    Args:
        a: first int
        b: second int
    """
    return a + b


def main():
    model = get_google_model(name="gemini-2.5-flash-preview-05-20")

    tools = [modo, multiply, add]

    tool_dict = {
        "modo": modo,
        "multiply": multiply,
        "add": add,
    }

    model = model.bind_tools(tools)

    messages = [
        SystemMessage(
            "You are a helpful assistant and every your response ends with an exclamation mark."
        ),
        HumanMessage(
            "What is 2 multiplied by 3? And by the way, what is the modo of 4 and 6?",
        ),
    ]

    response = model.invoke(messages)
    messages.append(response)

    if response.tool_calls is not None:
        for tool_call in response.tool_calls:
            tool = tool_dict[tool_call["name"].lower()]
            tool_message = tool.invoke(tool_call)
            messages.append(tool_message)

    response = model.invoke(messages)

    print(response)


if __name__ == "__main__":
    main()
