import os
import tqdm
import asyncio

from utility import *
from time import sleep
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

    # embeds = embedding_model.embed_documents(
    #     ["Lorem ipsum dolor sit amet, consectetur adipiscing elit."],
    #     output_dimensionality=3072
    # )

    # print(embeds[0])
    # return

    vector_store = InMemoryVectorStore(embedding_model)
    # Monkey-patch the vector store object to specify the embedding dimension and indent
    monkey_patch_in_memory_vector_store(vector_store)

    pages = load_pdf(pdf_path)

    # print(pages[10])

    splitter = CharacterTextSplitter(chunk_size=512, chunk_overlap=128)
    chunks = splitter.split_documents(pages)

    print(f"Total chunks: {len(chunks)}")
    print(f"First chunk length: {len(chunks[0].page_content)}")

    # print(chunks[10])

    for chunk in tqdm.tqdm(chunks, desc="Embedding chunks ......"):
        vector_store.add_documents(documents=[chunk], output_dimensionality=3072)  # Need TUN mode to go through proxy
        sleep(1)  # Avoid overwheling the experimental API

    docs = vector_store.similarity_search(
        "On April 13, 1867, what's the speed and location of Scotia?", k=2
    )
    for doc in docs:
        print("#" * 0x7F)
        print(doc)
        print()

    vector_store.dump(r"TTLUTS-First-8-Chapters-vector-store.json", indent=None)


if __name__ == "__main__":
    main()
