# ShadowMind — Architectural Decisions Log

Every significant technical choice is documented here with rationale.

---

## DECISION 001: FastAPI over Flask/Django

**Date:** 2026-03-14
**Context:** We need a Python web framework for our API server.
**Choice:** FastAPI
**Why:**
- **Async-native** — FastAPI is built on async Python, which we need for WebSocket streaming later (Week 7). Flask would require bolt-on async support.
- **Automatic API docs** — FastAPI generates interactive docs at `/docs` for free. This helps us test endpoints without Postman.
- **Pydantic integration** — FastAPI uses Pydantic for request/response validation natively. Since ShadowMind relies heavily on Pydantic models (Hypothesis, EvidenceChunk, etc.), this means zero glue code.
- **Performance** — FastAPI is one of the fastest Python frameworks, comparable to Node.js.
**Trade-off:** Smaller community than Flask/Django. But for an API-first AI project, FastAPI is the industry standard in 2025+.

---

## DECISION 002: Groq (Llama 3.3 70B) as Primary LLM

**Date:** 2026-03-14
**Context:** We need an LLM API that is free, fast, and supports structured outputs.
**Choice:** Groq API with Llama 3.3 70B
**Why:**
- **Free tier** — 30 req/min, 6000 tokens/min is enough for development and demos.
- **Speed** — Groq runs on custom LPU hardware. Response times are ~10x faster than OpenAI, which matters for real-time streaming.
- **Open model** — Llama 3.3 is open-source, no vendor lock-in.
- **Structured output support** — Groq supports JSON mode, which we need for Pydantic model generation.
**Backup:** Gemini 2.0 Flash (free tier) as fallback if Groq is rate-limited.

---

## DECISION 003: Pydantic v2 for Data Validation + Structured LLM Outputs

**Date:** 2026-03-17
**Context:** The LLM returns raw text by default. Our pipeline nodes need structured data with guaranteed fields and types.
**Choice:** Pydantic v2 BaseModel + Groq JSON mode (`response_format={"type": "json_object"}`)
**Why:**
- **Type safety** — Every field in Hypothesis, EvidenceChunk, etc. is validated at creation time. Wrong types → instant error, not a silent downstream bug.
- **Auto-documentation** — FastAPI reads Pydantic models to generate interactive API docs with example values.
- **Serialization** — `.model_dump()` converts any model to a dict/JSON instantly. No manual serialization code.
- **Field constraints** — `ge=0.0, le=1.0` on scores means Pydantic rejects invalid values before they enter the pipeline.
- **JSON mode** — Groq's `json_object` format forces the LLM to return valid JSON, eliminating parsing failures from stray text.
**Trade-off:** The LLM prompt must describe the exact JSON schema, which adds prompt tokens (~200 extra tokens per call). Acceptable cost for guaranteed structure.

---

## DECISION 004: Qdrant Cloud + Cohere for Vector Embeddings

**Date:** 2026-03-19
**Context:** ShadowMind needs to store evidence and search it by *meaning* (vector similarity). After Gemini's Google SDK persistently blocked the user's region/key with 404 errors, we briefly pivoted to local `sentence-transformers`, but the user requested a high-quality, high-dimension cloud API.
**Choice:** Qdrant Cloud (Cloud DB) + Cohere `embed-english-v3.0` API (Embedding Model).
**Why:**
- **Best-in-Class Quality** — Cohere's v3 embeddings are currently ranked as some of the highest-performing models for English semantic search, outputting huge 1024-dimensional vectors.
- **Extremely Generous Free Tier** — Cohere offers 1,000 requests per minute completely free, which is more than enough for our entire investigation pipeline.
- **Zero Region/Key Bugs** — Unlike the current Google genai SDK, Cohere's python SDK is rock solid and doesn't throw arbitrary 404s.
- **Qdrant Cloud Free Tier** — We kept Qdrant because it gives us 1GB of managed cloud storage for free, which easily holds our massive 1024-D vectors.
