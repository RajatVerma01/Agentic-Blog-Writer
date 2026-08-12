WRITER_SYSTEM_PROMPT = """You are a professional blog writer who produces high-quality, \
engaging, and well-researched articles.

Your task is to write a complete blog post in Markdown format following a provided outline exactly.

WRITING RULES:
1. Start with a single H1 heading (# Title) matching the outline title.
2. Write each section as an H2 heading (## Section Name) in the exact order given.
3. Cover every key point listed for each section — do not skip any.
4. Cite sources using the format: (Source: [Title](URL)) after any specific fact or statistic.
5. End with a ## Conclusion section summarizing the main takeaways.
6. Use clear, engaging language — avoid jargon unless explained.
7. Target the specified word count per section as closely as possible.
8. Do NOT add sections not in the outline.
9. Return ONLY the Markdown blog post — no explanations, no preamble."""


WRITER_USER_PROMPT = """Write a complete blog post using the outline and research below.

TOPIC: {topic}

OUTLINE:
Title: {title}
Total word target: {total_word_target} words

Sections:
{sections_text}

RESEARCH SOURCES (cite these in the blog):
{research_summary}

Write the full blog post in Markdown now."""


WRITER_REVISION_PROMPT = """You wrote a blog post that was evaluated and needs improvement.

TOPIC: {topic}

PREVIOUS DRAFT:
{draft}

{evaluation_context}

REVISION INSTRUCTIONS:
- Fix EVERY specific issue listed above.
- Keep all sections that were rated well.
- Do NOT change the overall structure unless structure_flow was flagged.
- Return ONLY the complete revised Markdown blog post."""


def format_sections_for_prompt(sections: list[dict]) -> str:
    """
    Converts the list of OutlineSection dicts into a numbered, readable
    text block for the Writer's LLM prompt.

    Args:
        sections: state["outline"]["sections"] — list of OutlineSection dicts.

    Returns:
        Formatted multi-line string the Writer LLM reads as instructions.
    """
    lines = []
    for i, sec in enumerate(sections, 1):
        key_points = "\n".join(f"    - {p}" for p in sec.get("key_points", []))
        lines.append(
            f"{i}. {sec.get('heading', f'Section {i}')} "
            f"(~{sec.get('word_target', 200)} words)\n{key_points}"
        )
    return "\n\n".join(lines)
