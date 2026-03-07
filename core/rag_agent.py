from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, SystemMessage
from core.vector_store import get_vectorstore

def ask(question: str, slug: str, history: list[dict]) -> str:
    vectorstore = get_vectorstore(slug)

    @tool(response_format="content_and_artifact")
    def retrieve_context(query: str):
        """Retrieve relevant excerpts from indexed research papers to answer a query."""
        retrieved_docs = vectorstore.similarity_search(query, k=5)
        serialized = "\n\n".join(
            f"Title: {doc.metadata.get('title', '?')}\n"
            f"Authors: {doc.metadata.get('authors', '?')}\n"
            f"Published: {doc.metadata.get('published', '?')}\n"
            f"Content: {doc.page_content}"
            for doc in retrieved_docs
        )
        return serialized, retrieved_docs

    llm = ChatOllama(model="qwen3.5:9b", temperature=0)
    prompt = (
        "You are an expert research assistant specialized in academic papers. "
        "Use the retrieve_context tool to find relevant information from indexed papers. "
        "Always cite the paper title and authors in your answer. "
        "Answer in the same language as the user's question."
    )

    agent = create_agent(llm, [retrieve_context], system_prompt=prompt)

    # Reconstruit les messages avec l'historique
    messages = []
    for msg in history[:-1]:  # Tout sauf la dernière question
        cls = HumanMessage if msg["role"] == "user" else SystemMessage
        messages.append(cls(content=msg["content"]))
    messages.append(HumanMessage(content=question))

    result = agent.invoke({"messages": messages})
    return result["messages"][-1].content