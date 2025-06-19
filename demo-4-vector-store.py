import os
import tqdm
import asyncio

import tqdm.asyncio

from utility import *
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter


def load_pdf(path):
    pages = []
    loader = PyPDFLoader(path)

    for page in tqdm.tqdm(loader.load(), desc="Loading PDF ......"):
        pages.append(page)

    return pages


def main():
    pdf_path = r"TTLUTS-First-8-Chapters.pdf"

    embedding_model = get_google_embedding_model()

    vector_store = InMemoryVectorStore(embedding_model)

    pages = load_pdf(pdf_path)

    print(pages[10])

    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(pages)

    print(f"Total chunks: {len(chunks)}")

    print(chunks[10])

    vector_store.add_documents(chunks[9:12])

    print(f"Total vectors: {len(vector_store)}")

    docs = vector_store.similarity_search("On April 13, 1867, what's the speed and location of Scotia?", k=2)
    for doc in docs:
        print(doc)
        print()


if __name__ == "__main__":
    main()
