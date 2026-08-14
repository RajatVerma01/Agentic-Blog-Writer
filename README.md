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
├── docker-compose.yml            # Multi-service orchestration
└── README.md                     # Developer onboarding + deployment guide
```

---

## 🚀 Quick Start — Local Development

### Prerequisites
- Python 3.11+
- A free [Groq API Key](https://console.groq.com/) (LLM inference)
- A free [Tavily API Key](https://tavily.com/) (web search)
- (Optional) A free [LangSmith API Key](https://smith.langchain.com/) (observability)

### 1. Clone and Set Up Environment

```bash
# Clone the repository
git clone https://github.com/your-username/agentic-blog-writer.git
cd agentic-blog-writer/backend

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate       # Mac/Linux
# venv\Scripts\activate        # Windows

# Install all dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Variables

```bash
# Copy the example file
cp .env.example .env
```

Open `.env` and fill in your API keys:

```env
# ----- LLM (Groq) -----
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL_NAME=llama-3.3-70b-versatile

# ----- Search (Tavily) -----
TAVILY_API_KEY=your_tavily_api_key_here

# ----- Observability (LangSmith) — Optional -----
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_api_key_here
LANGCHAIN_PROJECT=Agentic-Blog-Writer
```

### 3. Run the Server

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Open your browser at **[http://localhost:8000](http://localhost:8000)** and start generating blogs!

---

## ⚙️ All Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `GROQ_API_KEY` | ✅ Yes | — | Groq API key for LLM inference |
| `GROQ_MODEL_NAME` | No | `llama-3.3-70b-versatile` | Groq model to use |
| `GROQ_MAX_TOKENS` | No | `4096` | Max tokens per LLM response |
| `GROQ_TEMPERATURE` | No | `0.3` | LLM temperature (0.0–1.0) |
| `TAVILY_API_KEY` | ✅ Yes | — | Tavily Search API key |
| `TAVILY_MAX_RESULTS` | No | `5` | Max search results per query |
| `TAVILY_SEARCH_DEPTH` | No | `advanced` | Tavily search depth |
| `LANGCHAIN_TRACING_V2` | No | `false` | Enable LangSmith tracing |
| `LANGCHAIN_API_KEY` | No | — | LangSmith API key |
| `LANGCHAIN_PROJECT` | No | `default` | LangSmith project name |
| `APP_ENV` | No | `development` | `development` or `production` |
| `APP_HOST` | No | `0.0.0.0` | Host address to bind |
| `APP_PORT` | No | `8000` | Port to bind |
| `LOG_LEVEL` | No | `INFO` | Logging level |
| `MAX_REVISION_CYCLES` | No | `3` | Max writer/evaluator revision loops |
| `EVALUATION_THRESHOLD` | No | `7.0` | Minimum score (out of 10) to approve |
| `RESEARCHER_MAX_RESULTS` | No | `5` | Max sources the researcher returns |
| `TOOL_TIMEOUT_SECONDS` | No | `30` | Timeout for external tool calls |
| `RATE_LIMIT_GENERATE` | No | `5/minute` | Rate limit for `/generate` endpoint |
| `RATE_LIMIT_STATUS` | No | `30/minute` | Rate limit for `/status` endpoint |
| `JOB_TTL_HOURS` | No | `24` | How long completed jobs stay in memory |
| `BLOG_MIN_WORDS` | No | `500` | Minimum word count for output validation |
| `BLOG_MAX_WORDS` | No | `5000` | Maximum word count for output validation |
| `BLOCKED_KEYWORDS` | No | `[...]` | JSON list of prohibited input keywords |

---

## 🌐 API Reference

### `POST /api/v1/blog/generate`
Starts a new async blog generation job.

**Request Body:**
```json
{ "topic": "The Future of Artificial Intelligence in Healthcare" }
```

**Response:**
```json
{
  "job_id": "a1b2c3d4-...",
  "status": "queued",
  "message": "Blog generation started. Poll /api/v1/blog/status/... for updates."
}
```

---

### `GET /api/v1/blog/status/{job_id}`
Polls the real-time status of a running job.

**Response:**
```json
{
  "job_id": "a1b2c3d4-...",
  "status": "running",
  "current_agent": "writer",
  "created_at": "2026-08-13T16:00:00Z",
  "updated_at": "2026-08-13T16:00:12Z"
}
```

| `status` value | Meaning |
|---|---|
| `queued` | Job accepted, waiting for pipeline to start |
| `running` | Agents are actively generating the blog |
| `completed` | Blog is ready to fetch from `/result` |
| `failed` | An error occurred; check `error` field |

---

### `GET /api/v1/blog/result/{job_id}`
Fetches the final blog and evaluation scores once `status == "completed"`.

**Response:**
```json
{
  "job_id": "a1b2c3d4-...",
  "status": "completed",
  "topic": "The Future of AI in Healthcare",
  "final_blog": "# The Future of AI...\n\n## Introduction\n...",
  "evaluation_summary": {
    "score": 8.75,
    "approved": true,
    "scores_by_dimension": {
      "grammar_clarity": 9.0,
      "factual_accuracy": 8.5,
      "citation_quality": 8.0,
      "structure_flow": 9.0,
      "seo_optimization": 8.5
    },
    "improvements": ["Add more specific statistics", "..."]
  }
}
```

---

### `GET /api/v1/health`
Liveness probe for load balancers and deployment platforms.

```json
{ "status": "ok", "version": "1.0.0", "env": "development" }
```

---

## 🐳 Docker Deployment

### Run with Docker Compose (Recommended)

```bash
# From the backend/ directory
docker compose up -d --build
```

The app will be live at **http://localhost:8000**.

**Useful Docker commands:**
```bash
docker compose logs -f        # Tail live logs
docker compose down           # Stop the container
docker compose restart        # Restart after .env changes
```

### Manual Docker Build

```bash
docker build -t agentic-blog-writer .
docker run -d -p 8000:8000 --env-file .env agentic-blog-writer
```

---

## ☁️ Deployment on Render (Free Tier)

1. **Push code to GitHub** — Make sure `.env` is in `.gitignore` and never committed.
2. **Create a new Web Service** on [render.com](https://render.com).
3. **Connect your GitHub repo**.
4. **Configure the service:**
   - **Environment:** Docker (Render will auto-detect the `Dockerfile`)
   - **Instance Type:** Free
5. **Add Environment Variables** — Copy your `.env` keys (without the `.env` file) directly into the Render dashboard's "Environment Variables" section.
6. **Deploy** — Click "Create Web Service". Render will build and launch your app.

> **Note:** The free tier on Render spins down after 15 minutes of inactivity. Your first request after a spin-down will take ~30 seconds to wake up the server.

---

## 🔭 Observability with LangSmith

LangSmith provides deep tracing of every LLM call, tool invocation, and agent state transition — without any code changes required.

### Setup
1. Create a free account at [smith.langchain.com](https://smith.langchain.com/).
2. Generate an API key from Settings.
3. Add these three lines to your `.env`:
   ```env
   LANGCHAIN_TRACING_V2=true
   LANGCHAIN_API_KEY=lsv2_pt_...
   LANGCHAIN_PROJECT=Agentic-Blog-Writer
   ```
4. Restart the server and generate a blog. The project will appear automatically in LangSmith.

### What you can see
- **Visual agent timeline** — exactly when each agent (Researcher → Planner → Writer → Evaluator) ran and for how long.
- **Full prompt/response logs** — every message sent to and received from Groq.
- **Token usage** — per-agent and total token counts for cost tracking.
- **Error traces** — full stack traces for any LLM or tool failures.

---

## 🛡️ Security & Guardrails

| Layer | Implementation |
|---|---|
| **Input validation** | Blocks prompt injection, PII, and configured `BLOCKED_KEYWORDS` |
| **Rate limiting** | `slowapi` — 5 requests/min for generation, 30/min for status polling |
| **Output validation** | Enforces word count limits, heading structure, and URL sanity checks |
| **Non-root Docker** | Container runs as `appuser` (UID 1000), not root |
| **Secret hygiene** | `.env` is in `.gitignore`; secrets are never hardcoded |

---

## 🧪 Running Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run only unit tests
pytest tests/unit/

# Run only integration tests (requires valid API keys)
pytest tests/integration/
```

---

## 🛠️ Tech Stack

| Layer | Technology | Why |
|---|---|---|
| **LLM** | Groq (`llama-3.3-70b-versatile`) | Free tier, 300+ tokens/sec inference |
| **Agent Orchestration** | LangGraph 0.2 | Stateful, cyclical agent graphs |
| **LLM Abstraction** | LangChain 0.3 | Unified tool & prompt interface |
| **Web Search** | Tavily Search API | 1,000 free API calls/month |
| **Encyclopedic Facts** | Wikipedia API | Free, unlimited |
| **Backend Framework** | FastAPI 0.115 | Async-first, auto-docs, Pydantic |
| **Server** | Uvicorn | ASGI production server |
| **Frontend** | Vanilla HTML/CSS/JS | No build step, pure performance |
| **Rate Limiting** | slowapi | Per-IP limiting middleware |
| **Observability** | LangSmith | LLM trace & cost visibility |
| **Containerization** | Docker + Compose | Reproducible deployments |
| **Deployment** | Render (free) | Zero-cost cloud hosting |

---

## 🤝 Contributing

1. Fork the repository.
2. Create a feature branch: `git checkout -b feature/my-new-feature`
3. Make your changes, add tests where appropriate.
4. Submit a pull request with a clear description of your changes.

---

