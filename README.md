# Research Assistant

Streamlit app for searching, scoring, and indexing academic papers, with a RAG-based Q&A interface.

## Features

- Search papers from **ArXiv** and **Semantic Scholar**
- LLM-based relevance scoring per paper
- PDF download and indexing into a local vector store (ChromaDB)
- Q&A over indexed papers via RAG
- Theme-based organisation (one vector store per research topic)

## Architecture

```
app.py                   # Streamlit entry point
core/
  research_agent.py      # Search + scoring pipeline (LangChain + Ollama)
  rag_agent.py           # Q&A agent over indexed papers
  paper_fetcher.py       # PDF download
  vector_store.py        # ChromaDB wrapper
  theme_manager.py       # Theme persistence (JSON)
  logger.py              # Loguru setup
  sources/
    arxiv.py             # ArXiv source
    semantic_scholar.py  # Semantic Scholar source (with rate-limit retry)
tabs/
  bibliography.py        # Search & scoring tab
  papers.py              # Indexed papers tab
  qa.py                  # Q&A tab
data/                    # PDFs and ChromaDB, organised by theme slug
```

## Prerequisites

- [Ollama](https://ollama.com) running locally
- [uv](https://docs.astral.sh/uv/) for dependency management

Required Ollama models:

```bash
ollama pull qwen3.5:9b          # search & Q&A
ollama pull qwen2.5:3b          # paper scoring
ollama pull nomic-embed-text    # embeddings
```

## Installation

```bash
uv sync
```

## Running

```bash
uv run streamlit run app.py
```

### MCP server

`mcp_server.py` exposes the research functions (themes, search, indexing, Q&A) as an MCP server, e.g. for Claude Code:

```bash
claude mcp add research-tool -- /path/to/research_tool/mcp_server.sh
```

## Configuration

Copy `.env.example` to `.env` and fill in the values.

| Environment variable | Description | Default |
|---|---|---|
| `SEMANTIC_SCHOLAR_API_KEY` | Semantic Scholar API key (optional, increases rate limits) | — |

## Pipeline

**Phase 1 — Search** (`qwen3.5:9b`): the agent generates 2-3 optimised queries and calls the configured source tools (ArXiv, Semantic Scholar).

**Phase 2 — Scoring** (`qwen2.5:3b`): each paper is scored 1-10 for relevance to the research subject. Papers below the minimum score threshold are discarded.

**Indexing**: accepted PDFs are downloaded and chunked into ChromaDB via `nomic-embed-text` embeddings.

**Q&A**: the RAG agent retrieves relevant chunks and answers questions using `qwen3.5:9b`.

## Code quality

```bash
uv run ruff check .       # lint
uv run ruff format .      # format
```