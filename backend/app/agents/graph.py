from functools import lru_cache

from langgraph.graph import StateGraph, END

from app.agents.state import BlogState
from app.agents.researcher.agent import researcher_node
from app.agents.planner.agent import planner_node
from app.agents.writer.agent import writer_node
from app.agents.evaluator.agent import evaluator_node
from app.config.settings import get_settings
from app.utils.logger import get_logger

settings = get_settings()
logger = get_logger(__name__)


def _route_after_evaluation(state: BlogState) -> str:
    """
    Conditional edge function called after evaluator_node completes.

    Determines the next step based on 3 conditions checked in priority order:

    1. Error in state     → END  (don't loop on broken state)
    2. Blog approved      → END  (final_blog is set, pipeline done)
    3. Revision limit hit → END  (safety guard, return best draft so far)
    4. Otherwise          → "writer"  (another revision cycle)

    Args:
        state: The current full BlogState after evaluator_node ran.

    Returns:
        "writer" to trigger a revision, or END to finish the pipeline.
    """
    if state.get("error"):
        logger.warning(
            "Routing to END — error in state",
            extra={"error": state["error"]},
        )
        return END

    eval_result = state.get("evaluation_result", {})

    if eval_result.get("approved", False):
        logger.info(
            "Routing to END — blog approved",
            extra={"score": eval_result.get("score")},
        )
        return END

    if state["revision_count"] >= settings.MAX_REVISION_CYCLES:
        logger.warning(
            "Routing to END — max revision cycles reached",
            extra={
                "revision_count": state["revision_count"],
                "max": settings.MAX_REVISION_CYCLES,
                "score": eval_result.get("score"),
            },
        )
        return END

    logger.info(
        "Routing to writer — revision needed",
        extra={
            "score": eval_result.get("score"),
            "revision_count": state["revision_count"],
        },
    )
    return "writer"


def _build_graph() -> StateGraph:
    """
    Constructs and compiles the LangGraph pipeline.

    Graph topology:
        START → researcher → planner → writer → evaluator
                                           ↑          │
                                           │    approved=False
                                           └──── (revision loop)
                                                      │
                                               approved=True
                                               or error
                                               or max revisions
                                                      │
                                                     END

    Returns:
        A compiled LangGraph graph ready for ainvoke().
    """
    graph = StateGraph(BlogState)

    # Register all 4 agent nodes
    graph.add_node("researcher", researcher_node)
    graph.add_node("planner",    planner_node)
    graph.add_node("writer",     writer_node)
    graph.add_node("evaluator",  evaluator_node)

    # Fixed edges — always execute in this order
    graph.set_entry_point("researcher")
    graph.add_edge("researcher", "planner")
    graph.add_edge("planner",    "writer")
    graph.add_edge("writer",     "evaluator")

    # Conditional edge after evaluator — approve or revise
    graph.add_conditional_edges(
        "evaluator",
        _route_after_evaluation,
        {
            "writer": "writer",
            END:      END,
        },
    )

    logger.info("LangGraph pipeline compiled successfully")
    return graph.compile()


@lru_cache(maxsize=1)
def get_graph():
    """
    Returns the compiled LangGraph graph singleton.

    @lru_cache ensures the graph is built and compiled EXACTLY ONCE
    at startup, regardless of how many requests call get_graph().
    Graph compilation is expensive — this avoids per-request overhead.

    Used by blog_service.py:
        graph = get_graph()
        result = await graph.ainvoke(initial_state)

    In tests, call get_graph.cache_clear() to force a fresh compile.
    """
    return _build_graph()
