# 🤖 Agentic Blog Writer
### Complete Production-Ready Implementation

> A multi-agent blog writing system where users provide a topic and four specialized AI agents collaborate to produce a high-quality, fact-checked blog post. Built with **LangGraph** (stateful agent orchestration), **LangChain** (LLM abstraction + tools), **FastAPI** (backend), and a modern HTML/CSS/JS frontend. All resources are **100% free-tier compatible**.

---

## 🏗️ System Architecture — How It Works

```
User Input (Topic)
       │
       ▼
┌─────────────────────────────────────────────────────┐
│              LangGraph Orchestrator                  │
│                                                     │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐      │
│  │Researcher│───▶│ Planner  │───▶│  Writer  │      │
│  └──────────┘    └──────────┘    └──────────┘      │
│                                        │            │
│                                        ▼            │
│                                  ┌──────────┐       │
│                                  │Evaluator │       │
│                                  └────┬─────┘       │
│                                       │             │
│                        ┌─────────────┤             │
│                        ▼             ▼             │
│                   [APPROVED]    [REVISE]──▶Writer   │
└─────────────────────────────────────────────────────┘
       │
       ▼
   Final Blog Post (returned to User)
```

---

## 🤝 Agent Responsibilities

| Agent | Role | Tools Used |
|---|---|---|
| **Researcher** | Web search, fact gathering, source collection | Tavily Search API (free), Wikipedia |
| **Planner** | Outline creation, section structure, SEO keyword planning | LLM reasoning |
| **Writer** | Draft blog content following the plan, citing sources | LLM + context from planner |
| **Evaluator** | Check grammar, factual accuracy, citations, readability | LLM + Guardrails |

---

## 📁 Folder Structure — Backend

```
backend/
│
├── app/                          # Main application package
│   │
│   ├── __init__.py               # Package initializer
│   │
│   ├── main.py                   # FastAPI app entry point — registers all routers,
│   │                             # middlewares, CORS, exception handlers
│   │
│   ├── config/
│   │   ├── __init__.py
│   │   └── settings.py           # Pydantic BaseSettings — loads all env vars
│   │                             # (API keys, model names, rate limits).
│   │                             # Single source of truth for all configuration.
│   │
│   ├── api/                      # HTTP layer (FastAPI routers)
│   │   ├── __init__.py
│   │   ├── v1/
│   │   │   ├── __init__.py
│   │   │   ├── router.py         # Aggregates all v1 routes
│   │   │   └── endpoints/
│   │   │       ├── blog.py       # POST /blog/generate — starts blog generation
│   │   │       │                 # GET  /blog/status/{job_id} — poll job status
│   │   │       │                 # GET  /blog/result/{job_id} — fetch result
│   │   │       └── health.py     # GET  /health — liveness + readiness probe
│   │   └── dependencies.py       # FastAPI dependency injection (auth, rate limit)
│   │
│   ├── agents/                   # 🧠 CORE — All LangGraph agents live here
│   │   ├── __init__.py
│   │   │
│   │   ├── state.py              # BlogState TypedDict — the shared state object
│   │   │                         # that flows through the entire LangGraph graph.
│   │   │                         # Contains: topic, research_data, outline,
│   │   │                         # draft, evaluation_result, revision_count,
│   │   │                         # final_blog, messages, metadata
│   │   │
│   │   ├── graph.py              # LangGraph StateGraph definition — connects
│   │   │                         # all agent nodes, defines edges and conditional
│   │   │                         # routing (e.g., evaluator → writer if revise)
│   │   │
│   │   ├── researcher/
│   │   │   ├── __init__.py
│   │   │   ├── agent.py          # Researcher agent node function
│   │   │   ├── tools.py          # Tavily search tool, Wikipedia tool wrappers
│   │   │   └── prompts.py        # Researcher system prompt + few-shot examples
│   │   │
│   │   ├── planner/
│   │   │   ├── __init__.py
│   │   │   ├── agent.py          # Planner agent node function
│   │   │   └── prompts.py        # Planner system prompt for outline generation
│   │   │
│   │   ├── writer/
│   │   │   ├── __init__.py
│   │   │   ├── agent.py          # Writer agent node function
│   │   │   └── prompts.py        # Writer system prompt + revision instructions
│   │   │
│   │   └── evaluator/
│   │       ├── __init__.py
│   │       ├── agent.py          # Evaluator agent node function
│   │       ├── rubric.py         # Evaluation rubric (grammar, citations,
│   │       │                     # readability score, factual checks)
│   │       └── prompts.py        # Evaluator system prompt
│   │
│   ├── guardrails/               # 🛡️ Safety & Security Layer
│   │   ├── __init__.py
│   │   ├── input_validator.py    # Validates user input — blocks prompt injection,
│   │   │                         # checks topic length, blocked keywords, PII
│   │   ├── output_validator.py   # Validates agent outputs — prevents hallucinated
│   │   │                         # citations, blocks harmful content in output
│   │   └── rate_limiter.py       # In-memory rate limiter per IP (slowapi)
│   │
│   ├── tools/                    # Reusable LangChain tool definitions
│   │   ├── __init__.py
│   │   ├── search.py             # Tavily Search API wrapper (free 1000 calls/mo)
│   │   ├── wikipedia.py          # Wikipedia tool for factual grounding
│   │   └── text_utils.py         # Word count, readability scoring helpers
│   │
│   ├── schemas/                  # Pydantic models for API request/response
│   │   ├── __init__.py
│   │   ├── blog.py               # BlogRequest, BlogResponse, JobStatus models
│   │   └── evaluation.py         # EvaluationResult, ImprovementPoint models
│   │
│   ├── services/                 # Business logic layer (between API and agents)
│   │   ├── __init__.py
│   │   └── blog_service.py       # Orchestrates job queuing, runs LangGraph graph,
│   │                             # stores results in background task
│   │
│   ├── storage/                  # Lightweight persistence (no paid DB needed)
│   │   ├── __init__.py
│   │   └── job_store.py          # In-memory job store (thread-safe dict with
│   │                             # TTL cleanup). Can be swapped to Redis/SQLite.
│   │
│   └── utils/
│       ├── __init__.py
│       ├── logger.py             # Structured logging (Python logging + JSON format)
│       └── exceptions.py         # Custom exception classes + global handler
│
├── tests/                        # Test suite
│   ├── __init__.py
│   ├── conftest.py               # Pytest fixtures (mock LLM, mock search)
│   ├── unit/
│   │   ├── test_guardrails.py    # Test input/output validation
│   │   ├── test_state.py         # Test state transitions
│   │   └── test_schemas.py       # Test Pydantic models
│   └── integration/
│       └── test_graph.py         # Test full LangGraph pipeline (mocked LLM)
│
├── .env.example                  # Template for environment variables (no secrets)
├── .env                          # Actual secrets — NEVER commit this (in .gitignore)
├── .gitignore                    # Git ignore patterns
├── requirements.txt              # Python dependencies with pinned versions
├── Dockerfile                    # Docker container definition for deployment
├── docker-compose.yml            # Multi-service orchestration (app + optional redis)
└── README.md                     # Developer onboarding + deployment guide
```
