from utility import *

from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

def modo(a, b):
    return a * a * b + b * b * a

@tool
def modo_operation(a: int, b: int) -> int:
    """Compute the modo of two integers.

    Args:
        a (int): The first integer.
        b (int): The second integer.
    """

    return modo(a, b)


def main():
    model = get_google_model(name="gemini-2.5-flash-preview-05-20")

    tools = [modo_operation]

    model = model.bind_tools(tools)

    messages = [
        SystemMessage(
            "You are a helpful assistant and every your response ends with an exclamation mark."
        ),
        HumanMessage(
            "What is the modo of 3 and 4?",
        ),
    ]
    # need to do with an agent object

    response = model.invoke(messages)
    print(response.content)
    print("////")
    print(response)


if __name__ == "__main__":
    main()

    
