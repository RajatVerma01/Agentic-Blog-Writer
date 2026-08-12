
PLANNER_SYSTEM_PROMPT = """You are an expert content strategist and blog architect.

Your job is to take a blog topic and research data, then produce a clear,
structured blog outline that a Writer agent will follow precisely.

RULES YOU MUST FOLLOW:
1. The outline must have 4 to 6 sections (not counting the intro).
2. Each section must have 2 to 5 specific key points the writer must cover.
3. Key points must directly reference facts from the provided research data.
4. The blog title must be SEO-friendly and compelling (max 70 characters).
5. Include 5 to 8 SEO keywords naturally relevant to the topic.
6. Total word target must be between 800 and 2000 words.
7. Each section word target must add up to the total.

OUTPUT FORMAT:
Return ONLY valid JSON matching this exact structure. No markdown, no explanation:
{
    "title": "SEO-friendly blog title here",
    "sections": [
        {
            "heading": "Section H2 heading",
            "key_points": ["point 1", "point 2", "point 3"],
            "word_target": 250
        }
    ],
    "keywords": ["keyword1", "keyword2", "keyword3"],
    "total_word_target": 1200
}"""


PLANNER_USER_PROMPT = """Create a detailed blog outline for the following topic using the research below.

Topic: {topic}

Research Data:
{research_summary}

Produce a structured outline with 4-6 sections. Each section must have specific
key points grounded in the research data above.

Return ONLY valid JSON. No extra text."""


def format_research_for_prompt(research_data: list[dict]) -> str:
    """
    Converts the list of ResearchSource dicts from BlogState into a
    compact, readable text block for the Planner's LLM prompt.

    Kept in prompts.py (not agent.py) because it is prompt-formatting
    logic — its purpose is to prepare input for the LLM prompt.

    Args:
        research_data: state["research_data"] — list of ResearchSource dicts.

    Returns:
        Formatted multi-line string with numbered sources.
    """
    if not research_data:
        return "No research data available."

    lines = []
    for i, source in enumerate(research_data, 1):
        lines.append(f"[{i}] {source.get('title', 'Untitled')}")
        lines.append(f"    URL: {source.get('url', 'N/A')}")
        lines.append(f"    Excerpt: {source.get('snippet', '')[:400]}")
        lines.append("")

    return "\n".join(lines).strip()
