import json
import re
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage
from core.logger import logger
from core.sources import AVAILABLE_SOURCES
from core.sources.base import NormalizedPaper

SEARCH_SYSTEM_PROMPT = """You are an academic paper search specialist. You have ONE job: search for papers.

STRICT RULES:
- Generate 2-3 optimized search queries in English using academic terms
- Call the available search tools for each query
- If multiple search tools are available, use each of them for better coverage
- If the subject is too vague, ask exactly ONE clarifying question in the user's language
- Never comment on results, never summarize, never score papers
- After all searches are done, respond with exactly: SEARCH_COMPLETE"""

SCORING_SYSTEM_PROMPT = """You are an academic paper relevance evaluator.

You will receive a research subject and a list of papers. Score each paper.

STRICT RULES:
- Score every paper from 1 to 10 based on relevance to the research subject
- relevance_score MUST be an integer between 1 and 10
- Provide a brief justification in the user's language
- Respond ONLY with a valid JSON array, nothing else

REQUIRED FORMAT:
[
  {
    "paper_id": "id_from_the_input_list",
    "title": "Paper title",
    "authors": "Author 1, Author 2",
    "published": "2023",
    "abstract": "Abstract...",
    "relevance_score": 8,
    "justification": "Pertinent car..."
  }
]

IMPORTANT: Use the exact paper_id from the input list. Do not invent or modify IDs."""


def _extract_papers_from_response(content: str) -> list[dict] | None:
    candidates = []

    for match in re.finditer(r'```json\s*(\[.*?\])\s*```', content, re.DOTALL):
        try:
            parsed = json.loads(match.group(1))
            if isinstance(parsed, list):
                candidates.append(parsed)
        except json.JSONDecodeError:
            pass

    for match in re.finditer(r'\[\s*\{.*?\}\s*\]', content, re.DOTALL):
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, list):
                candidates.append(parsed)
        except json.JSONDecodeError:
            pass

    if not candidates:
        return None

    ranked = sorted(
        candidates,
        key=lambda c: sum(1 for p in c if "relevance_score" in p or "score" in p),
        reverse=True
    )
    best = ranked[0]
    if not any("relevance_score" in p or "score" in p for p in best):
        return None
    return best


def _normalize_from_cache(paper: dict, active_sources: list) -> NormalizedPaper | None:
    """Reconstruit un NormalizedPaper depuis les caches des sources actives."""
    paper_id = paper.get("paper_id") or paper.get("arxiv_id") or paper.get("id")
    if not paper_id:
        return None

    for source_name in active_sources:
        source = AVAILABLE_SOURCES.get(source_name)
        if not source:
            continue
        normalized = source.get_paper(paper_id)
        if normalized:
            # Injecte les champs du scoring
            normalized.relevance_score = int(
                paper.get("relevance_score") or paper.get("score") or 0
            )
            normalized.justification = paper.get("justification", "—")
            return normalized

    return None


def _run_search_phase(
    messages: list,
    existing_paper_ids: set[str],
    active_sources: list[str],
    steps_callback=None
) -> tuple[str | None, list[dict], list]:

    sources = [AVAILABLE_SOURCES[s] for s in active_sources if s in AVAILABLE_SOURCES]
    tools = [s.tool for s in sources]

    if not tools:
        logger.error("Aucun tool de recherche disponible")
        return None, [], messages

    llm = ChatOllama(model="qwen3.5:9b", temperature=0)
    llm_with_tools = llm.bind_tools(tools)

    existing_ids_block = ""
    if existing_paper_ids:
        ids_list = "\n".join(f"- {pid}" for pid in existing_paper_ids)
        existing_ids_block = f"\n\nALREADY INDEXED - DO NOT retrieve these:\n{ids_list}"

    sources_block = "\nAvailable search tools: " + ", ".join(s.tool.name for s in sources)
    system_prompt = SEARCH_SYSTEM_PROMPT + sources_block + existing_ids_block

    original_subject = next(
        (m.content for m in messages if isinstance(m, HumanMessage)),
        "the research subject"
    )

    full_messages = [SystemMessage(content=system_prompt)] + [
        m for m in messages if not isinstance(m, SystemMessage)
    ]

    updated_messages = list(full_messages)
    collected_papers: list[dict] = []

    # Map tool_name → source pour le dispatch
    tool_map = {s.tool.name: s.tool for s in sources}

    while True:
        response = llm_with_tools.invoke(updated_messages)
        updated_messages.append(response)

        if response.tool_calls:
            for tc in response.tool_calls:
                tool_name = tc["name"]
                logger.info(f"Recherche [{tool_name}] : query='{tc['args'].get('query')}'")

                if steps_callback:
                    steps_callback({"type": "tool_call", "tool": tool_name, "args": tc["args"]})

                tool_fn = tool_map.get(tool_name)
                if not tool_fn:
                    logger.warning(f"Tool inconnu : {tool_name}")
                    continue

                tool_result = tool_fn.invoke(tc["args"])

                try:
                    results = json.loads(tool_result)
                    collected_papers.extend(results)
                    logger.info(f"{len(results)} papiers via {tool_name} (total: {len(collected_papers)})")
                    if steps_callback:
                        steps_callback({"type": "tool_result", "count": len(results), "source": tool_name})
                except Exception:
                    pass

                updated_messages.append(ToolMessage(
                    content=f"[REMINDER: Searching for: {original_subject}]\n\n{tool_result}",
                    tool_call_id=tc["id"],
                    name=tool_name
                ))
        else:
            content = response.content.strip()
            if "SEARCH_COMPLETE" in content or collected_papers:
                logger.success(f"Phase 1 terminée — {len(collected_papers)} papiers collectés")
                return None, collected_papers, updated_messages
            return content, [], updated_messages


def _run_scoring_phase(
    subject: str,
    raw_papers: list[dict],
    active_sources: list[str],
    min_score: int = 5
) -> list[NormalizedPaper]:

    llm = ChatOllama(model="qwen3.5:9b", temperature=0)

    papers_for_scoring = json.dumps([
        {
            "paper_id": p.get("id", p.get("arxiv_id", p.get("paper_id"))),
            "title": p.get("t", p.get("title")),
            "authors": p.get("a", p.get("authors")),
            "published": p.get("y", p.get("published")),
            "abstract": p.get("s", p.get("abstract"))
        }
        for p in raw_papers
    ], ensure_ascii=False, separators=(",", ":"))

    scoring_messages = [
        SystemMessage(content=SCORING_SYSTEM_PROMPT),
        HumanMessage(content=(
            f"Research subject: {subject}\n\n"
            f"Papers to score:\n{papers_for_scoring}\n\n"
            f"Return ONLY the JSON array."
        ))
    ]

    logger.debug(f"=== Phase 2 : Scoring de {len(raw_papers)} papiers ===")
    response = llm.invoke(scoring_messages)
    scored_papers = _extract_papers_from_response(response.content.strip())
    logger.debug(f"Réponse brute du scoring : {response.content.strip()[:500]}")

    if not scored_papers:
        logger.warning("Échec scoring, tentative correction")
        correction = llm.invoke(scoring_messages + [HumanMessage(
            content=f"Invalid format. Return ONLY the JSON array. Subject: {subject}"
        )])
        logger.debug(f"Réponse de correction : {correction.content.strip()[:500]}")
        scored_papers = _extract_papers_from_response(correction.content.strip())

    if not scored_papers:
        logger.error("Échec total du scoring")
        return []

    enriched = []
    for paper in scored_papers:
        normalized = _normalize_from_cache(paper, active_sources)
        if not normalized:
            continue

        try:
            score = int(normalized.relevance_score)
        except (ValueError, TypeError):
            score = 0

        if score >= min_score:
            normalized.relevance_score = score
            enriched.append(normalized)

    enriched.sort(key=lambda p: p.relevance_score, reverse=True)
    logger.success(f"Phase 2 — {len(enriched)}/{len(scored_papers)} papiers retenus (score ≥ {min_score})")
    return enriched


def run_research_agent_turn(
    messages: list,
    existing_paper_ids: set[str] = None,
    min_score: int = 5,
    sources: list[str] = None,
    steps_callback=None,
    fresh_start: bool = False
) -> tuple[str | None, list[NormalizedPaper], list]:

    active_sources = sources or ["arxiv"]

    if fresh_start:
        for source_name in active_sources:
            source = AVAILABLE_SOURCES.get(source_name)
            if source:
                source.clear_cache()
        logger.info(f"Caches vidés pour : {active_sources}")

    if steps_callback:
        steps_callback({
            "type": "phase",
            "label": f"🔍 Phase 1 : Recherche sur {', '.join(active_sources)}..."
        })

    clarification, raw_papers, updated_messages = _run_search_phase(
        messages,
        existing_paper_ids or set(),
        active_sources,
        steps_callback=steps_callback
    )

    if clarification:
        return clarification, [], updated_messages

    if not raw_papers:
        logger.error("Aucun papier trouvé")
        return None, [], updated_messages

    if steps_callback:
        steps_callback({
            "type": "phase",
            "label": f"🧠 Phase 2 : Scoring de {len(raw_papers)} papiers..."
        })

    original_subject = next(
        (m.content for m in messages if isinstance(m, HumanMessage)),
        "the research subject"
    )

    scored = _run_scoring_phase(original_subject, raw_papers, active_sources, min_score)
    return None, scored, updated_messages