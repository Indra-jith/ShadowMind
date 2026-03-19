# ShadowMind 🕵️

**What they don't want you to find.**

ShadowMind is an autonomous investigation agent. Give it a mystery, conspiracy, or unexplained event — and it will generate competing hypotheses, find evidence for and against each one, eliminate the weak theories, and present you with a surviving conclusion backed by a full evidence trail.

## What Makes It Different

Standard RAG: "What documents match this query?"
ShadowMind: "What evidence would **break** this hypothesis?"

That inversion — hypothesis first, evidence second — is what makes it generative rather than retrievive.

## Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM | Groq (Llama 3.3 70B) |
| Embeddings | Gemini Embedding API |
| Vector DB | Qdrant Cloud |
| Knowledge Graph | NetworkX |
| Agent Framework | LangGraph |
| Web Search | Tavily |
| Backend | FastAPI + WebSockets |
| Frontend | React + Vite |
| Observability | Langfuse |
| Evaluation | DeepEval |

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/shadowmind.git
cd shadowmind

# 2. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables
copy .env.example .env
# Edit .env and add your GROQ_API_KEY

# 5. Run the server
uvicorn backend.api.main:app --reload

# 6. Open your browser
# http://localhost:8000       — health check
# http://localhost:8000/docs  — interactive API docs
```

## Project Status

🚧 Week 1 — Building the foundation.
