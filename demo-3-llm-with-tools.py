from utility import *

from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage


@tool
def modo_operation(input1: str, input2: str) -> str:
    """Apply modo operation to two input strings.

    Args:
        input1 (str): First input string.
        input2 (str): Second input string.
    """

    return f"eye [{input1}] eye | nose | eye [{input2}] eye"


def main():
    model = get_google_model(name="gemini-2.0-flash")

    tools = [modo_operation]

    model = model.bind_tools(tools)

    messages = [
        SystemMessage(
            "You are a helpful assistant and every your response ends with an exclamation mark."
        ),
        HumanMessage(
            # "What is 'hihi' modo 'hoho'?",
            "Tell me a joke",
        ),
    ]
    # need to do with an agent object

    response = model.invoke(messages)
    print(response.content)
    print("////")
    print(response)


if __name__ == "__main__":
    main()
