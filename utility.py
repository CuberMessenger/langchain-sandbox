import os
import getpass

from langchain_google_genai import ChatGoogleGenerativeAI


def get_google_api_key():
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google API Key: ")
    return os.environ["GOOGLE_API_KEY"]


def get_google_cse_id():
    os.environ["GOOGLE_CSE_ID"] = getpass.getpass("Enter your Google CSE ID: ")
    return os.environ["GOOGLE_CSE_ID"]

def get_google_model(name="gemini-2.0-flash"):
    api_key = get_google_api_key()

    model = ChatGoogleGenerativeAI(
        model=name,
        google_api_key=api_key,
        transport="rest",
        timeout=30,
        max_retries=0,
    )

    return model
