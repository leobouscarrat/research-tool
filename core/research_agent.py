import json
import re

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from core.logger import logger
from core.sources import AVAILABLE_SOURCES
from core.sources.base import NormalizedPaper

SEARCH_MODEL = "qwen3.5:9b"
SCORING_MODEL = "qwen2.5:3b"

MAX_SEARCH_TOOL_CALLS = 8

PLANNING_SYSTEM_PROMPT = """You are an academic research query specialist.

Analyze the user's research subject and output search queries.

RULES:
- If the subject is too vague or ambiguous, respond with ONE clarifying question \
in the user's language, prefixed with CLARIFICATION:
- Otherwise, output ONLY a JSON array of search queries in English using precise academic terminology
- Each query will be run on {source_count} source(s) — stay within a total budget of {max_tool_calls} searches
- Queries should cover different angles: core concept, methods, applications, recent advances
- No explanation, no preamble — only the JSON array or the CLARIFICATION: line

Examples of valid outputs:
["transformer attention mechanism survey", "self-attention NLP sequence modeling", "BERT GPT pretraining language model"]
CLARIFICATION: Cherches-tu des papiers sur l'aspect théorique ou les applications industrielles ?"""

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


def _run_query_planning_phase(messages: list, active_sources: list[str]) -> tuple[str | None, list[str]]:
    """
    Phase 0 : construit les requêtes de recherche à partir du sujet utilisateur.
    Retourne (clarification, []) si le sujet est trop vague,
    ou (None, queries) avec les requêtes prêtes à exécuter.
    """
    llm = ChatOllama(model=SEARCH_MODEL, temperature=0)
    system_prompt = PLANNING_SYSTEM_PROMPT.format(
        source_count=len(active_sources),
        max_tool_calls=MAX_SEARCH_TOOL_CALLS,
    )
    planning_messages = [SystemMessage(content=system_prompt)] + [
        m for m in messages if not isinstance(m, SystemMessage)
    ]

    response = llm.invoke(planning_messages)
    content = response.content.strip()

    if content.startswith("CLARIFICATION:"):
        question = content[len("CLARIFICATION:"):].strip()
        logger.info("Phase 0 : clarification demandée")
        return question, []

    # Parse le tableau JSON de requêtes
    try:
        match = re.search(r'\[.*\]', content, re.DOTALL)
        if match:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, list) and parsed:
                # Normalise : aplatit les éventuelles sous-listes, filtre les non-strings
                flat: list[str] = []
                for item in parsed:
                    if isinstance(item, str) and item.strip():
                        flat.append(item.strip())
                    elif isinstance(item, list):
                        flat.extend(s.strip() for s in item if isinstance(s, str) and s.strip())
                if flat:
                    logger.info(f"Phase 0 : {len(flat)} requêtes planifiées : {flat}")
                    return None, flat
    except (json.JSONDecodeError, ValueError):
        pass

    # Fallback : utilise le message brut de l'utilisateur
    logger.warning("Phase 0 : parsing échoué, utilisation du sujet brut comme requête")
    original = next((m.content for m in messages if isinstance(m, HumanMessage)), "")
    return None, [original] if original else []


def _run_search_phase(
    queries: list[str],
    existing_paper_ids: set[str],
    active_sources: list[str],
    steps_callback=None,
) -> list[dict]:
    """
    Phase 1 : exécute les requêtes sur chaque source de manière programmatique.
    Retourne la liste des papiers collectés (sans doublons, sans déjà indexés).
    """
    sources = [AVAILABLE_SOURCES[s] for s in active_sources if s in AVAILABLE_SOURCES]
    if not sources:
        logger.error("Aucune source de recherche disponible")
        return []

    collected_papers: list[dict] = []
    seen_paper_ids: set[str] = set(existing_paper_ids)  # exclut les déjà indexés
    tool_call_count = 0

    for query in queries:
        for source in sources:
            if tool_call_count >= MAX_SEARCH_TOOL_CALLS:
                logger.warning(f"Limite de {MAX_SEARCH_TOOL_CALLS} appels atteinte, recherche arrêtée")
                break

            logger.info(f"Recherche [{source.tool.name}] : query='{query}'")
            if steps_callback:
                steps_callback({"type": "tool_call", "tool": source.tool.name, "args": {"query": query}})

            tool_result = source.tool.invoke({"query": query})
            tool_call_count += 1

            try:
                results = json.loads(tool_result)
                new_papers = [p for p in results if p.get("id") not in seen_paper_ids]
                seen_paper_ids.update(p.get("id") for p in new_papers)
                collected_papers.extend(new_papers)
                dupes = len(results) - len(new_papers)
                logger.info(
                    f"{len(new_papers)} papiers via {source.tool.name}"
                    + (f" ({dupes} doublons ignorés)" if dupes else "")
                    + f" (total: {len(collected_papers)})"
                )
                if steps_callback:
                    steps_callback({"type": "tool_result", "count": len(new_papers), "source": source.tool.name})
            except Exception as e:
                logger.debug(f"Parsing résultat [{source.tool.name}] ignoré : {e}")

        if tool_call_count >= MAX_SEARCH_TOOL_CALLS:
            break

    logger.success(f"Phase 1 terminée — {len(collected_papers)} papiers collectés")
    return collected_papers


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
        steps_callback({"type": "phase", "label": "🧭 Phase 0 : Planification des requêtes..."})

    clarification, queries = _run_query_planning_phase(messages, active_sources)

    if clarification:
        return clarification, [], messages

    if steps_callback:
        steps_callback({
            "type": "phase",
            "label": f"🔍 Phase 1 : Recherche sur {', '.join(active_sources)}..."
        })

    raw_papers = _run_search_phase(queries, existing_paper_ids or set(), active_sources, steps_callback)

    if not raw_papers:
        logger.error("Aucun papier trouvé")
        return None, [], messages

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
    return None, scored, messages
