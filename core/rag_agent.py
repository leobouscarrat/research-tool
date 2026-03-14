from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama

from core.logger import logger
from core.vector_store import get_vectorstore

LLM_MODEL = "qwen3.5:9b"

RAG_SYSTEM_PROMPT = """You are an expert research assistant specialized in academic papers.

STRICT RULES:
- Always use the retrieve_context tool to find relevant information before answering
- You may call retrieve_context multiple times with different queries if needed
- Always cite the paper title and authors for every claim you make
- Answer in the same language as the user's question
- Be precise and structured in your answer"""


def ask(question: str, slug: str, history: list[dict]) -> str:
    vectorstore = get_vectorstore(slug)

    @tool
    def retrieve_context(query: str) -> str:
        """Retrieve relevant excerpts from indexed research papers to answer a query."""
        retrieved_docs = vectorstore.similarity_search(query, k=5)
        if not retrieved_docs:
            return "No relevant documents found."
        serialized = "\n\n".join(
            f"Title: {doc.metadata.get('title', '?')}\n"
            f"Authors: {doc.metadata.get('authors', '?')}\n"
            f"Published: {doc.metadata.get('published', '?')}\n"
            f"Content: {doc.page_content}"
            for doc in retrieved_docs
        )
        logger.info(f"RAG : {len(retrieved_docs)} chunks récupérés pour '{query}'")
        return serialized

    llm = ChatOllama(model=LLM_MODEL, temperature=0)
    llm_with_tools = llm.bind_tools([retrieve_context])
    tool_map = {"retrieve_context": retrieve_context}

    # Reconstruit les messages avec l'historique
    messages = [SystemMessage(content=RAG_SYSTEM_PROMPT)]
    for msg in history[:-1]:  # Tout sauf la dernière question
        cls = HumanMessage if msg["role"] == "user" else AIMessage
        messages.append(cls(content=msg["content"]))
    messages.append(HumanMessage(content=question))

    # Boucle manuelle identique à research_agent.py
    max_iterations = 5
    for i in range(max_iterations):
        response = llm_with_tools.invoke(messages)
        messages.append(response)

        if not response.tool_calls:
            # Pas d'appel outil → réponse finale
            logger.success(f"RAG : réponse générée après {i + 1} itération(s)")
            return response.content

        # Exécute les tool calls
        for tc in response.tool_calls:
            tool_fn = tool_map.get(tc["name"])
            if not tool_fn:
                logger.warning(f"RAG : tool inconnu '{tc['name']}'")
                continue
            logger.info(f"RAG : retrieve_context query='{tc['args'].get('query', '')}'")
            result = tool_fn.invoke(tc["args"])
            messages.append(ToolMessage(
                content=result,
                tool_call_id=tc["id"],
                name=tc["name"]
            ))

    logger.warning("RAG : max itérations atteint, retourne le dernier contenu")
    return response.content
