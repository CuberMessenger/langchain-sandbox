import os
import getpass

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage


def main():
    if not os.environ.get("GOOGLE_API_KEY"):
        os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google API Key: ")

    api_key = os.environ["GOOGLE_API_KEY"]
    print(f"Key head: {api_key[:4]}...")
    print(f"Key tail: ...{api_key[-4:]}")

    model = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=api_key,
        transport="rest",  # force HTTP/REST
        timeout=30,  # give up after 30 seconds
        max_retries=0,  # don’t retry on failure
    )

    messages = [
        SystemMessage("You are a nordic poet."),
        HumanMessage("Write a rhyming poem about C# the programming language."),
    ]

    try:
        response = model.invoke(messages)
        print(response.content)
    except Exception as e:
        print("Error or timeout:", e)


if __name__ == "__main__":
    main()
