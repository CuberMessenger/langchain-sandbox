import os
import pickle
import getpass

from langchain_google_genai import ChatGoogleGenerativeAI


class LocalEnvironment:
    ENVIRONMENT_DATA_FILE = "env.pkl"

    GOOGLE_API_KEY = None
    GOOGLE_CSE_ID = None

    @staticmethod
    def prepare_environment():

        data = {
            "GOOGLE_API_KEY": getpass.getpass("Enter your Google API Key: "),
            "GOOGLE_CSE_ID": getpass.getpass("Enter your Google CSE ID: "),
        }

        with open(LocalEnvironment.ENVIRONMENT_DATA_FILE, "wb") as file:
            pickle.dump(data, file)

    @staticmethod
    def load_environment():
        if not os.path.exists(LocalEnvironment.ENVIRONMENT_DATA_FILE):
            LocalEnvironment.prepare_environment()

        with open(LocalEnvironment.ENVIRONMENT_DATA_FILE, "rb") as file:
            data = pickle.load(file)

        LocalEnvironment.GOOGLE_API_KEY = data["GOOGLE_API_KEY"]
        LocalEnvironment.GOOGLE_CSE_ID = data["GOOGLE_CSE_ID"]

    @staticmethod
    def get_google_api_key():
        if LocalEnvironment.GOOGLE_API_KEY is None:
            LocalEnvironment.load_environment()
        return LocalEnvironment.GOOGLE_API_KEY
    
    @staticmethod
    def get_google_cse_id():
        if LocalEnvironment.GOOGLE_CSE_ID is None:
            LocalEnvironment.load_environment()
        return LocalEnvironment.GOOGLE_CSE_ID


def get_google_model(name="gemini-2.0-flash"):
    api_key = LocalEnvironment.get_google_api_key()

    model = ChatGoogleGenerativeAI(
        model=name,
        google_api_key=api_key,
        transport="rest",
        timeout=30,
        max_retries=3,
    )

    return model
