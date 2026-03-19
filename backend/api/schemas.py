"""
shadowmind/backend/api/schemas.py

Responsibility: Defines the request and response shapes for our API endpoints.
These are SEPARATE from the investigation models because:
  - Investigation models = internal pipeline contracts (used between nodes)
  - API schemas = external contracts (what the client sends/receives)

Keeping them separate means we can change the API format without touching
the pipeline logic, and vice versa.

Used by: backend/api/main.py
Depends on: pydantic, backend/models/investigation.py
"""

# ============================================================
# IMPORTS
# ============================================================

# BaseModel — parent class for Pydantic models (explained in investigation.py)
from pydantic import BaseModel, Field

# List — type hint for lists (explained in investigation.py)
from typing import List

# Import our investigation models — WHY: the API response includes
# Hypothesis objects, so we need access to the Hypothesis class.
from backend.models.investigation import Hypothesis


# ============================================================
# REQUEST SCHEMAS — what the client sends TO our API
# ============================================================

class InvestigateRequest(BaseModel):
    """
    The request body for POST /investigate.

    WHY a Pydantic model instead of a raw string parameter:
    1. Validates that the mystery is not empty
    2. The request is now a JSON body (standard for APIs) instead of
       a query parameter (which has URL length limits)
    3. We can add more fields later (e.g., depth, max_hypotheses)
       without changing the endpoint signature

    Example request body:
    {
        "mystery": "What happened at Dyatlov Pass?"
    }
    """

    # mystery — the user's question / unexplained event to investigate.
    # min_length=10 prevents garbage inputs like "hi" or "??"
    mystery: str = Field(
        ...,
        min_length=10,
        max_length=1000,
        description="The mystery or unexplained event to investigate",
        examples=["What happened to the hikers at Dyatlov Pass in 1959?"],
    )


# ============================================================
# RESPONSE SCHEMAS — what our API sends BACK to the client
# ============================================================

class InvestigateResponse(BaseModel):
    """
    The response body for POST /investigate.

    WHY a Pydantic model instead of a raw dict:
    1. FastAPI uses this to generate accurate API docs automatically
    2. It guarantees the response always has the same shape — the
       frontend can rely on "hypotheses" always being a list
    3. If we accidentally forget to include a field, Pydantic catches
       it before the response is sent

    Example response:
    {
        "mystery": "What happened at Dyatlov Pass?",
        "hypotheses": [...],
        "model": "llama-3.3-70b-versatile"
    }
    """

    # mystery — echoed back so the frontend knows which question was asked.
    mystery: str = Field(..., description="The original mystery that was investigated")

    # hypotheses — the structured list of generated hypotheses.
    # WHY a List[Hypothesis] and not raw text: each hypothesis is a validated
    # Pydantic object with guaranteed fields (id, title, score, status).
    # The frontend can directly render these without parsing text.
    hypotheses: List[Hypothesis] = Field(
        ...,
        description="Generated hypotheses for the mystery",
    )

    # model — which LLM model was used (for transparency/debugging).
    model: str = Field(..., description="LLM model used for generation")
