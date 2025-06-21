import os
import pickle
import getpass

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings


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


def get_google_embedding_model(name="models/text-embedding-004"):
    """
    models/gemini-embedding-exp-03-07
    models/text-embedding-004
    """
    api_key = LocalEnvironment.get_google_api_key()

    embedding_model = GoogleGenerativeAIEmbeddings(
        model=name,
        google_api_key=api_key,
        transport="rest",
        timeout=30,
        max_retries=3,
    )

    return embedding_model


def monkey_patch_in_memory_vector_store(in_memory_vector_store):
    ### patch 1: add output_dimensionality parameter to add_documents method
    import uuid
    import types

    from langchain_core.documents import Document
    from typing import Any, Optional
    from collections.abc import Iterator

    def add_documents_patched(
        self,
        documents: list[Document],
        ids: Optional[list[str]] = None,
        output_dimensionality: Optional[int] = None,  # This line is added
        **kwargs: Any,
    ) -> list[str]:
        """Add documents to the store."""
        texts = [doc.page_content for doc in documents]
        vectors = self.embedding.embed_documents(
            texts, output_dimensionality=output_dimensionality
        )  # This line is modified

        if ids and len(ids) != len(texts):
            msg = (
                f"ids must be the same length as texts. "
                f"Got {len(ids)} ids and {len(texts)} texts."
            )
            raise ValueError(msg)

        id_iterator: Iterator[Optional[str]] = (
            iter(ids) if ids else iter(doc.id for doc in documents)
        )

        ids_ = []

        for doc, vector in zip(documents, vectors):
            doc_id = next(id_iterator)
            doc_id_ = doc_id or str(uuid.uuid4())
            ids_.append(doc_id_)
            self.store[doc_id_] = {
                "id": doc_id_,
                "vector": vector,
                "text": doc.page_content,
                "metadata": doc.metadata,
            }

        return ids_

    in_memory_vector_store.add_documents = types.MethodType(
        add_documents_patched, in_memory_vector_store
    )
    ### patch 1: add output_dimensionality parameter to add_documents method

    ### patch 2: add indent to dump method
    import json
    from pathlib import Path
    from langchain_core.load import dumpd

    def dump_patched(self, path: str, indent: Optional[int] = 2) -> None:
        """Dump the vector store to a file.

        Args:
            path: The path to dump the vector store to.
        """
        _path: Path = Path(path)
        _path.parent.mkdir(exist_ok=True, parents=True)
        with _path.open("w") as f:
            json.dump(dumpd(self.store), f, indent=indent)

    in_memory_vector_store.dump = types.MethodType(dump_patched, in_memory_vector_store)
    ### patch 2: add indent to dump method
