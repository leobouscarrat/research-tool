from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

def get_vectorstore(slug: str):
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return Chroma(
        persist_directory=f"./data/themes/{slug}/chroma_db",
        embedding_function=embeddings
    )