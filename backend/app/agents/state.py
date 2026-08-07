
from typing import Annotated, Optional, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class ResearchSource(TypedDict):
    """
    A single research source found by the Researcher agent.
    Stored in state["research_data"] as a list of these dicts.

    Example:
        {
            "title":   "AI in Healthcare — Stanford Study 2023",
            "url":     "https://stanford.edu/ai-health-2023",
            "snippet": "AI diagnostic tools achieved 94% accuracy...",
            "source":  "tavily"
        }
    """
    title:   str   # Title of the article or page
    url:     str   # Source URL (used for citation validation in output guardrail)
    snippet: str   # Key extracted text (what the writer will use as evidence)
    source:  str   # "tavily" | "wikipedia" | "manual"


class OutlineSection(TypedDict):
    """
    A single section of the blog outline produced by the Planner agent.

    Example:
        {
            "heading":    "How AI Diagnoses Diseases",
            "key_points": [
                "AI image recognition in radiology",
                "ML models for cancer detection",
                "94% accuracy statistic from Stanford study"
            ],
            "word_target": 300
        }
    """
    heading:     str        # H2 heading text for this section
    key_points:  list[str]  # Bullet points the Writer must cover in this section
    word_target: int        # Approximate word count the Writer should aim for


class BlogOutline(TypedDict):
    """
    The complete blog structure produced by the Planner agent.
    Stored in state["outline"].

    Example:
        {
            "title":    "The Future of AI in Healthcare",
            "sections": [ OutlineSection, OutlineSection, ... ],
            "keywords": ["AI healthcare", "machine learning diagnosis", ...],
            "total_word_target": 1200
        }
    """
    title:             str                  # SEO-optimized blog title
    sections:          list[OutlineSection] # Ordered list of sections
    keywords:          list[str]            # SEO keywords to use naturally in text
    total_word_target: int                  # Target total word count


class EvaluationResultState(TypedDict):
    """
    A TypedDict version of the EvaluationResult for use inside BlogState.
    This mirrors the Pydantic model from schemas/evaluation.py but as a
    plain dict — LangGraph state cannot hold Pydantic objects directly.

    Populated by the Evaluator agent node.
    Read by the Writer agent node on revision cycles.

    Fields:
        score              Weighted score 0.0–10.0 (computed)
        approved           True if score >= 7.0
        dimension_scores   {"grammar_clarity": 8.0, "citation_quality": 4.0, ...}
        improvements       List of {dimension, issue, suggestion} dicts
        feedback           Paragraph summary for the Writer
        revision_number    Which revision this evaluation is for
    """
    score:            float
    approved:         bool
    dimension_scores: dict[str, float]
    improvements:     list[dict]   # List of ImprovementPoint dicts
    feedback:         str
    revision_number:  int


class JobMetadata(TypedDict):
    """
    System metadata about the current job — not used by agents directly,
    but stored in state so the blog_service can read it after the graph runs.

    Example:
        {
            "job_id":    "abc-123-xyz",
            "started_at": "2024-01-15T10:30:00Z",
            "topic_safe": True
        }
    """
    job_id:     str   # UUID from job_store — used for logging and status updates
    started_at: str   # ISO 8601 timestamp string
    topic_safe: bool  # True if input_validator passed — always True by the time
                      # state is created (guardrail blocks unsafe topics before this)



class BlogState(TypedDict):
    """
    The complete shared state that travels through the entire LangGraph graph.

    LangGraph rule: Every agent node receives the FULL state and returns a
    PARTIAL dict containing only the fields it changed. LangGraph merges
    the partial update back into the full state before passing to the next node.

    Example of a partial update returned by the researcher node:
        return {
            "research_data": [
                {"title": "...", "url": "...", "snippet": "...", "source": "tavily"},
                ...
            ]
        }
        # LangGraph merges this into the full BlogState automatically.
        # All other fields (topic, outline, draft, etc.) remain unchanged.

    Field initialization (set by blog_service before graph starts):
        topic:            set from user input
        research_data:    []  (empty, researcher fills it)
        outline:          {}  (empty, planner fills it)
        draft:            ""  (empty, writer fills it)
        evaluation_result: {} (empty, evaluator fills it)
        revision_count:   0
        final_blog:       None
        messages:         []
        metadata:         {job_id, started_at, topic_safe}
        error:            None
    """


    topic: str
    """
    The user-provided blog topic, sanitized by the input guardrail.
    Example: "The Future of Artificial Intelligence in Healthcare"
    Set by: blog_service.py before invoking the graph
    Read by: researcher, planner, writer (all agents need the topic)
    Never modified after initialization.
    """

    

    research_data: list[ResearchSource]
    """
    List of research sources gathered by the Researcher agent.
    Example: [
        {"title": "Stanford AI Study", "url": "https://...", "snippet": "..."},
        {"title": "WHO AI Report",     "url": "https://...", "snippet": "..."},
    ]
    Set by:  researcher_node
    Read by: planner_node, writer_node, evaluator_node
    """

    outline: BlogOutline
    """
    The blog structure created by the Planner agent.
    Contains: title, sections (headings + key points), keywords, word targets.
    Set by:  planner_node
    Read by: writer_node, evaluator_node
    """

    draft: str
    """
    The full blog post in Markdown format, written by the Writer agent.
    On revision cycles, the writer overwrites this with an improved version.
    Example: "# AI in Healthcare\n\nArtificial intelligence is revolutionizing..."
    Set by:  writer_node (overwritten on each revision)
    Read by: evaluator_node
    """

    evaluation_result: EvaluationResultState
    """
    The Evaluator agent's structured assessment of the current draft.
    Contains: score, approved, dimension_scores, improvements, feedback.
    Set by:  evaluator_node
    Read by: writer_node (on revision cycles — reads .improvements and .feedback)
    The conditional edge in graph.py reads evaluation_result["approved"] to
    decide whether to route back to writer or end the pipeline.
    """

    

    revision_count: int
    """
    How many writer → evaluator revision cycles have completed.
    Starts at 0 (first draft).
    Incremented by writer_node on each revision.
    The conditional edge enforces: if revision_count >= MAX_REVISION_CYCLES → END
    This is the safety guard that prevents infinite revision loops.
    """


    final_blog: Optional[str]
    """
    The approved final blog post in Markdown format.
    Set to state["draft"] by the evaluator_node when approved=True.
    None until the blog is approved.
    blog_service.py reads this after the graph finishes to store the result.
    """

    

    messages: Annotated[list[BaseMessage], add_messages]
    """
    The conversation message history used by agent LLM calls.

    The Annotated[..., add_messages] is a LangGraph reducer:
    - Without it: each agent REPLACES the entire messages list.
    - With it:    each agent APPENDS to the existing messages list.

    This preserves the full conversation context across all agents,
    which is essential for the Writer to understand the Evaluator's
    feedback and for the Evaluator to compare the revised draft.

    Populated by: each agent node (via LangChain LLM calls)
    """

   
    metadata: JobMetadata
    """
    Job-level metadata: job_id, started_at, topic_safe.
    Set once by blog_service.py before invoking the graph.
    Read after graph completion to build BlogMetadata for the API response.
    Agents do not write to this field.
    """

    error: Optional[str]
    """
    Set to an error message string if any agent raises an exception.
    None during normal operation.
    blog_service.py checks this after graph.ainvoke() to determine if
    the job succeeded or failed, and updates the job store accordingly.

    Example: "researcher agent failed: Tavily API timeout after 30s"
    """




def create_initial_state(topic: str, job_id: str) -> BlogState:
    """
    Creates and returns the initial BlogState for a new blog generation job.

    Called by blog_service.py before invoking the LangGraph graph:
        initial_state = create_initial_state(topic="AI in Healthcare", job_id="abc-123")
        result = await compiled_graph.ainvoke(initial_state)

    Args:
        topic:  The sanitized blog topic from the user.
        job_id: The UUID from the job store for this generation request.

    Returns:
        A fully initialized BlogState with all fields set to safe defaults.
    """
    from datetime import datetime, timezone

    return BlogState(
        # Core input
        topic=topic,

        # Agent outputs — all empty until filled by agents
        research_data=[],
        outline=BlogOutline(
            title="",
            sections=[],
            keywords=[],
            total_word_target=0,
        ),
        draft="",
        evaluation_result=EvaluationResultState(
            score=0.0,
            approved=False,
            dimension_scores={},
            improvements=[],
            feedback="",
            revision_number=0,
        ),

        # Control flow
        revision_count=0,

        # Final output
        final_blog=None,

        # Message history
        messages=[],

        # System metadata
        metadata=JobMetadata(
            job_id=job_id,
            started_at=datetime.now(tz=timezone.utc).isoformat(),
            topic_safe=True,  # True because guardrail already validated the topic
        ),

        # Error tracking
        error=None,
    )
