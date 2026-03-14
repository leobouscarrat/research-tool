import json
import re

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_ollama import ChatOllama

from core.logger import logger
from core.sources import AVAILABLE_SOURCES
from core.sources.base import NormalizedPaper

SEARCH_MODEL = "qwen3.5:9b"
SCORING_MODEL = "qwen2.5:3b"

MAX_SEARCH_TOOL_CALLS = 8

SEARCH_SYSTEM_PROMPT = """You are an academic paper search specialist. You have ONE job: search for papers.

STRICT RULES:
- Generate 2-3 optimized search queries in English using academic terms
- You have a budget of {max_tool_calls} tool calls maximum in total — use them wisely across all sources
- If multiple search tools are available, split your queries across them for better coverage
- If the subject is too vague, ask exactly ONE clarifying question in the user's language
- Never comment on results, never summarize, never score papers
- After all searches are done, respond with exactly: SEARCH_COMPLETE"""

SCORING_SYSTEM_PROMPT = """You are an academic paper relevance evaluator.

You will receive a research subject and a single paper. Score it.

STRICT RULES:
- Score the paper from 1 to 10 based on relevance to the research subject
- relevance_score MUST be an integer between 1 and 10
- Provide a brief justification in the user's language
- Respond ONLY with a valid JSON object, nothing else

REQUIRED FORMAT:
{
  "relevance_score": 8,
  "justification": "Pertinent car..."
}"""


def _get_paper_id(paper: dict) -> str | None:
    return paper.get("id") or paper.get("paper_id") or paper.get("arxiv_id")


def _extract_score_from_response(content: str) -> dict | None:
    """Extrait un objet JSON {relevance_score, justification} depuis la réponse du LLM."""
    # Tente d'abord un bloc ```json ... ```
    for match in re.finditer(r'```json\s*(\{.*?\})\s*```', content, re.DOTALL):
        try:
            parsed = json.loads(match.group(1))
            if "relevance_score" in parsed:
                return parsed
        except json.JSONDecodeError:
            pass

    # Tente un objet JSON brut
    for match in re.finditer(r'\{[^{}]*\}', content, re.DOTALL):
        try:
            parsed = json.loads(match.group(0))
            if "relevance_score" in parsed:
                return parsed
        except json.JSONDecodeError:
            pass

    return None


def _normalize_from_cache(paper: dict, active_sources: list) -> NormalizedPaper | None:
    """Reconstruit un NormalizedPaper depuis les caches des sources actives."""
    paper_id = _get_paper_id(paper)
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

    llm = ChatOllama(model=SEARCH_MODEL, temperature=0)
    llm_with_tools = llm.bind_tools(tools)

    existing_ids_block = ""
    if existing_paper_ids:
        ids_list = "\n".join(f"- {pid}" for pid in existing_paper_ids)
        existing_ids_block = f"\n\nALREADY INDEXED - DO NOT retrieve these:\n{ids_list}"

    sources_block = "\nAvailable search tools: " + ", ".join(s.tool.name for s in sources)
    system_prompt = (
        SEARCH_SYSTEM_PROMPT.format(max_tool_calls=MAX_SEARCH_TOOL_CALLS)
        + sources_block
        + existing_ids_block
    )

    original_subject = next(
        (m.content for m in messages if isinstance(m, HumanMessage)),
        "the research subject"
    )

    full_messages = [SystemMessage(content=system_prompt)] + [
        m for m in messages if not isinstance(m, SystemMessage)
    ]

    updated_messages = list(full_messages)
    collected_papers: list[dict] = []
    tool_call_count = 0

    # Map tool_name → source pour le dispatch
    tool_map = {s.tool.name: s.tool for s in sources}

    while True:
        response = llm_with_tools.invoke(updated_messages)
        updated_messages.append(response)

        if response.tool_calls:
            for tc in response.tool_calls:
                if tool_call_count >= MAX_SEARCH_TOOL_CALLS:
                    logger.warning(f"Limite de {MAX_SEARCH_TOOL_CALLS} appels atteinte, recherche arrêtée")
                    break

                tool_name = tc["name"]
                logger.info(f"Recherche [{tool_name}] : query='{tc['args'].get('query')}'")

                if steps_callback:
                    steps_callback({"type": "tool_call", "tool": tool_name, "args": tc["args"]})

                tool_fn = tool_map.get(tool_name)
                if not tool_fn:
                    logger.warning(f"Tool inconnu : {tool_name}")
                    continue

                tool_result = tool_fn.invoke(tc["args"])
                tool_call_count += 1

                try:
                    results = json.loads(tool_result)
                    collected_papers.extend(results)
                    logger.info(f"{len(results)} papiers via {tool_name} (total: {len(collected_papers)})")
                    if steps_callback:
                        steps_callback({"type": "tool_result", "count": len(results), "source": tool_name})
                except Exception as e:
                    logger.debug(f"Parsing résultat [{tool_name}] ignoré : {e}")

                updated_messages.append(ToolMessage(
                    content=f"[REMINDER: Searching for: {original_subject}]\n\n{tool_result}",
                    tool_call_id=tc["id"],
                    name=tool_name
                ))

            if tool_call_count >= MAX_SEARCH_TOOL_CALLS:
                logger.success(f"Phase 1 terminée (limite atteinte) — {len(collected_papers)} papiers collectés")
                return None, collected_papers, updated_messages
        else:
            content = response.content.strip()
            if "SEARCH_COMPLETE" in content or collected_papers:
                logger.success(f"Phase 1 terminée — {len(collected_papers)} papiers collectés")
                return None, collected_papers, updated_messages
            return content, [], updated_messages


def _score_single_paper(
    llm,
    subject: str,
    paper: dict,
) -> dict | None:
    """Score un seul papier. Retourne {relevance_score, justification} ou None."""
    paper_id = _get_paper_id(paper)
    abstract = paper.get("s", paper.get("abstract", ""))  # abstract complet

    scoring_messages = [
        SystemMessage(content=SCORING_SYSTEM_PROMPT),
        HumanMessage(content=(
            f"Research subject: {subject}\n\n"
            f"Paper to score:\n"
            f"Title: {paper.get('t', paper.get('title', ''))}\n"
            f"Authors: {paper.get('a', paper.get('authors', ''))}\n"
            f"Published: {paper.get('y', paper.get('published', ''))}\n"
            f"Abstract: {abstract}\n\n"
            f"Return ONLY the JSON object."
        ))
    ]

    response = llm.invoke(scoring_messages)
    result = _extract_score_from_response(response.content.strip())

    if not result:
        # Tentative de correction
        correction = llm.invoke(scoring_messages + [HumanMessage(
            content="Invalid format. Return ONLY a JSON object with relevance_score (int 1-10) and justification."
        )])
        result = _extract_score_from_response(correction.content.strip())

    if result:
        result["paper_id"] = paper_id
    return result


def _run_scoring_phase(
    subject: str,
    raw_papers: list[dict],
    active_sources: list[str],
    min_score: int = 5,
    steps_callback=None,
) -> list[NormalizedPaper]:

    total = len(raw_papers)
    logger.debug(f"=== Phase 2 : Scoring de {total} papiers (modèle: {SCORING_MODEL}) ===")

    llm = ChatOllama(model=SCORING_MODEL, temperature=0)
    enriched: list[NormalizedPaper] = []
    failed = 0

    for i, paper in enumerate(raw_papers):
        paper_id = _get_paper_id(paper)

        if steps_callback:
            steps_callback({"type": "scoring_progress", "current": i + 1, "total": total})

        scored = _score_single_paper(llm, subject, paper)
        if not scored:
            logger.warning(f"Échec scoring pour {paper_id}, papier ignoré")
            failed += 1
            continue

        try:
            score = int(scored.get("relevance_score", 0))
        except (ValueError, TypeError):
            score = 0

        if score < min_score:
            logger.debug(f"Score {score} < {min_score}, papier écarté : {paper_id}")
            continue

        normalized = _normalize_from_cache({"paper_id": paper_id, **scored}, active_sources)
        if not normalized:
            logger.warning(f"Papier introuvable en cache : {paper_id}")
            failed += 1
            continue

        normalized.relevance_score = score
        normalized.justification = scored.get("justification", "—")
        enriched.append(normalized)

    enriched.sort(key=lambda p: p.relevance_score, reverse=True)
    logger.success(
        f"Phase 2 — {len(enriched)}/{len(raw_papers)} papiers retenus "
        f"(score ≥ {min_score}, {failed} échecs)"
    )
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

    scored = _run_scoring_phase(original_subject, raw_papers, active_sources, min_score, steps_callback)
    return None, scored, updated_messages
