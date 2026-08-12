from app.schemas.evaluation import RubricDimension, RUBRIC_WEIGHTS


EVALUATOR_SYSTEM_PROMPT = """You are a strict, expert blog quality evaluator.

Your job is to score a blog post across 5 dimensions using a 0.0–10.0 scale.
You must return your evaluation as valid JSON — nothing else.

SCORING DIMENSIONS:
1. grammar_clarity    (weight 25%) — Grammar, spelling, sentence clarity, readability
2. factual_accuracy   (weight 25%) — Are facts correct and supported by the research sources?
3. citation_quality   (weight 20%) — Are specific claims cited with source URLs?
4. structure_flow     (weight 15%) — Logical progression, H1/H2 structure, conclusion present?
5. seo_optimization   (weight 15%) — Keywords used naturally, title under 70 chars, meta-friendly?

SCORING GUIDE:
  9.0–10.0 = Exceptional. Publish-ready with no changes needed.
  7.0–8.9  = Good. Minor issues, meets the approval threshold.
  5.0–6.9  = Fair. Clear weaknesses that need addressing.
  3.0–4.9  = Poor. Significant problems throughout.
  0.0–2.9  = Unacceptable. Requires complete rewrite.

IMPROVEMENT RULES:
- Only add an improvement point if the score for that dimension is below 7.0.
- Every improvement must name the exact location (paragraph, section) and exact fix.
- Maximum 10 improvement points total.
- If score >= 7.0 for all dimensions, improvements list must be empty.

OUTPUT FORMAT — return ONLY this JSON, no markdown fences, no extra text:
{
    "dimension_scores": [
        {"dimension": "grammar_clarity",  "score": 8.0, "rationale": "..."},
        {"dimension": "factual_accuracy", "score": 7.5, "rationale": "..."},
        {"dimension": "citation_quality", "score": 5.0, "rationale": "..."},
        {"dimension": "structure_flow",   "score": 8.5, "rationale": "..."},
        {"dimension": "seo_optimization", "score": 7.0, "rationale": "..."}
    ],
    "improvements": [
        {"dimension": "citation_quality", "issue": "exact issue", "suggestion": "exact fix"}
    ],
    "feedback": "2-4 sentence summary for the writer."
}"""


EVALUATOR_USER_PROMPT = """Evaluate the following blog post.

TOPIC: {topic}

BLOG POST TO EVALUATE:
{draft}

RESEARCH SOURCES AVAILABLE (used to check factual accuracy and citation quality):
{research_summary}

REVISION NUMBER: {revision_number}

Score each of the 5 dimensions on a 0.0–10.0 scale.
Return ONLY valid JSON. No explanation outside the JSON."""


def format_research_for_evaluation(research_data: list[dict]) -> str:
    """
    Formats research sources for the evaluator prompt so the LLM can
    cross-check facts in the draft against actual source content.

    Deliberately shorter format than format_research_for_prompt() in
    planner/prompts.py — the evaluator only needs source title + URL +
    a short excerpt to fact-check, not the full snippet.

    Args:
        research_data: state["research_data"] list of ResearchSource dicts.

    Returns:
        Compact multi-line string of sources for the evaluation prompt.
    """
    if not research_data:
        return "No research sources available."

    return "\n".join(
        f"- [{s.get('title', 'Untitled')}]({s.get('url', 'N/A')}): "
        f"{s.get('snippet', '')[:200]}"
        for s in research_data
    )
