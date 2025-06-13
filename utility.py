import os
import getpass

from langchain_google_genai import ChatGoogleGenerativeAI


"""
In powershell, you can set environment variables like this:

$Env:GOOGLE_API_KEY = "your_api_key"
$Env:GOOGLE_CSE_ID = "your_cse_id"
...
Clear-History
"""


def get_google_api_key():
    if os.environ["GOOGLE_API_KEY"] is None:
        os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google API Key: ")
    return os.environ["GOOGLE_API_KEY"]


def get_google_cse_id():
    if os.environ.get("GOOGLE_CSE_ID") is None:
        os.environ["GOOGLE_CSE_ID"] = getpass.getpass("Enter your Google CSE ID: ")
    return os.environ["GOOGLE_CSE_ID"]


def get_google_model(name="gemini-2.0-flash"):
    api_key = get_google_api_key()

    model = ChatGoogleGenerativeAI(
        model=name,
        google_api_key=api_key,
        transport="rest",
        timeout=30,
        max_retries=3,
    )

    return model
