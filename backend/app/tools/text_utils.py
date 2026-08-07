
import re
from urllib.parse import urlparse




def _strip_markdown(text: str) -> str:
    """
    Strips Markdown syntax from text for plain-text analysis.
    Used by count_words() and flesch_reading_ease() — extracted here
    to avoid duplicating the same 5 regex substitutions in both functions.
    """
    clean = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    clean = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", clean)
    clean = re.sub(r"`[^`]+`", " ", clean)
    clean = re.sub(r"```[\s\S]*?```", " ", clean)
    clean = re.sub(r"(\*{1,2}|_{1,2})([^*_]+)\1", r"\2", clean)
    clean = re.sub(r"^[-*_]{3,}\s*$", "", clean, flags=re.MULTILINE)
    clean = re.sub(r"https?://\S+", " ", clean)
    return clean


def count_words(text: str) -> int:
    """
    Counts the number of words in a text string.

    Strips Markdown syntax before counting so the count reflects
    actual content words, not markup.

    Args:
        text: Any string — plain text or Markdown.

    Returns:
        Integer word count. Returns 0 for empty/whitespace-only strings.
    """
    if not text or not text.strip():
        return 0
    clean = _strip_markdown(text)
    words = [w for w in clean.split() if w.strip()]
    return len(words)




def extract_urls(text: str) -> list[str]:
    """
    Extracts all http/https URLs from a text string.

    Finds:
    - Bare URLs: https://example.com/article
    - Markdown link URLs: [text](https://example.com)
    - URLs in parentheses, after colons, etc.

    Deduplicates results (preserves first-seen order).

    Args:
        text: Any string — plain text or Markdown.

    Returns:
        List of unique URL strings. Empty list if no URLs found.

    Examples:
        extract_urls("See https://example.com for details")
        → ["https://example.com"]

        extract_urls("[Stanford Study](https://stanford.edu/ai) and [WHO](https://who.int)")
        → ["https://stanford.edu/ai", "https://who.int"]
    """
    if not text:
        return []

    # Pattern matches http:// and https:// URLs
    # Stops at whitespace, closing parenthesis, comma, or period at end
    url_pattern = re.compile(
        r"https?://"           # scheme
        r"[a-zA-Z0-9\-._~:/?#\[\]@!$&'()*+,;=%]+"  # valid URL chars
    )

    found = url_pattern.findall(text)

    # Clean trailing punctuation that regex may have captured
    cleaned: list[str] = []
    for url in found:
        url = url.rstrip(".,;:!?)'\"")
        if url:
            cleaned.append(url)

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for url in cleaned:
        if url not in seen:
            seen.add(url)
            unique.append(url)

    return unique


def deduplicate_by_url(items: list[dict], url_key: str = "url") -> list[dict]:
    """
    Removes duplicate dicts from a list based on the URL field.
    Preserves first-seen order. Items with empty/missing URL are kept.

    Used by: researcher/tools.py (merge Tavily + Wikipedia results)
             tools/search.py (merge multi-query results)

    Args:
        items:   List of dicts, each expected to have a URL field.
        url_key: The dict key that holds the URL (default: "url").

    Returns:
        Deduplicated list preserving original order.

    Example:
        results = [{"url": "https://a.com", ...}, {"url": "https://a.com", ...}]
        deduplicate_by_url(results)  → [{"url": "https://a.com", ...}]
    """
    seen: set[str] = set()
    unique: list[dict] = []
    for item in items:
        url = item.get(url_key, "")
        if url and url in seen:
            continue
        if url:
            seen.add(url)
        unique.append(item)
    return unique


def is_valid_url(url: str) -> bool:
    """
    Checks if a string is a syntactically valid URL.
    Does NOT make a network request — only validates the format.

    Args:
        url: The URL string to check.

    Returns:
        True if the URL has a valid scheme and netloc, False otherwise.

    Examples:
        is_valid_url("https://example.com")   → True
        is_valid_url("not-a-url")             → False
        is_valid_url("http://")               → False
    """
    try:
        parsed = urlparse(url)
        return bool(parsed.scheme in {"http", "https"} and parsed.netloc)
    except Exception:
        return False



def flesch_reading_ease(text: str) -> float:
    """
    Calculates the Flesch Reading Ease score for a text.

    Target for blog posts: 50–70 (readable but substantive).
    Formula: 206.835 - 1.015 × (words/sentences) - 84.6 × (syllables/words)

    Args:
        text: Plain text or Markdown string.

    Returns:
        Flesch score as a float (0.0–100.0). Returns 0.0 for very short texts.
    """
    if not text or not text.strip():
        return 0.0

    clean = _strip_markdown(text)

    words = [w for w in clean.split() if w.strip()]
    word_count = len(words)
    if word_count == 0:
        return 0.0

    sentence_endings = re.findall(r"[.!?]+", clean)
    sentence_count = max(len(sentence_endings), 1)

    syllable_count = sum(_count_syllables(word) for word in words)

    avg_sentence_length = word_count / sentence_count
    avg_syllables_per_word = syllable_count / word_count

    score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
    return round(max(0.0, min(100.0, score)), 2)


def _count_syllables(word: str) -> int:
    """
    Estimates the number of syllables in a single English word.
    Uses a heuristic approach — not perfect, but accurate enough for
    readability scoring purposes.

    Args:
        word: A single word string (may contain punctuation).

    Returns:
        Estimated syllable count (minimum 1).
    """
    word = re.sub(r"[^a-zA-Z]", "", word).lower()
    if not word:
        return 1

    # Count vowel groups as syllable nuclei
    vowels = "aeiouy"
    count = 0
    prev_was_vowel = False

    for char in word:
        is_vowel = char in vowels
        if is_vowel and not prev_was_vowel:
            count += 1
        prev_was_vowel = is_vowel

    # Apply common English rules
    if word.endswith("e") and len(word) > 2:
        count -= 1   # Silent 'e' at end (e.g., "make" = 1 syllable, not 2)
    if word.endswith("le") and len(word) > 2:
        count += 1   # "-le" endings (e.g., "table" = 2 syllables)
    if word.endswith("es") and len(word) > 3:
        count -= 1   # Plural "-es" often silent

    return max(1, count)


#

def has_h1_heading(text: str) -> bool:
    """
    Checks if the text contains at least one H1 Markdown heading (# Heading).

    Used by output_validator to ensure the blog has a proper title.

    Args:
        text: Markdown string.

    Returns:
        True if at least one `# Heading` is found.
    """
    return bool(re.search(r"^#\s+\S", text, flags=re.MULTILINE))


def count_h2_headings(text: str) -> int:
    """
    Counts the number of H2 Markdown headings (## Heading) in the text.

    Used by output_validator to ensure the blog has proper section structure.
    Minimum expected: 3 H2 sections (intro body, main sections, conclusion).

    Args:
        text: Markdown string.

    Returns:
        Count of `## Heading` occurrences.
    """
    return len(re.findall(r"^##\s+\S", text, flags=re.MULTILINE))


def has_conclusion(text: str) -> bool:
    """
    Checks if the blog post contains a conclusion section.

    Looks for common conclusion heading patterns (case-insensitive):
    - ## Conclusion
    - ## Final Thoughts
    - ## Wrapping Up
    - ## Summary
    - ## Key Takeaways

    Args:
        text: Markdown string.

    Returns:
        True if a conclusion-like heading is found.
    """
    pattern = re.compile(
        r"^#{1,3}\s+(conclusion|final\s+thoughts?|wrapping\s+up|summary|key\s+takeaways?)",
        flags=re.MULTILINE | re.IGNORECASE,
    )
    return bool(pattern.search(text))


def get_text_stats(text: str) -> dict:
    """
    Returns a summary dict of text statistics in a single call.
    Used by the evaluator agent and output_validator.

    Args:
        text: Markdown blog post string.

    Returns:
        {
            "word_count":          1247,
            "sentence_count":      62,
            "h1_count":            1,
            "h2_count":            5,
            "url_count":           7,
            "has_conclusion":      True,
            "reading_ease_score":  58.3
        }
    """
    word_count = count_words(text)
    sentence_count = max(len(re.findall(r"[.!?]+", text)), 1)

    return {
        "word_count":         word_count,
        "sentence_count":     sentence_count,
        "h1_count":           len(re.findall(r"^#\s+\S", text, re.MULTILINE)),
        "h2_count":           count_h2_headings(text),
        "url_count":          len(extract_urls(text)),
        "has_conclusion":     has_conclusion(text),
        "reading_ease_score": flesch_reading_ease(text),
    }