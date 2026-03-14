from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings


def get_vectorstore(slug: str) -> Chroma:
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return Chroma(
        persist_directory=f"./data/themes/{slug}/chroma_db",
        embedding_function=embeddings
    )
