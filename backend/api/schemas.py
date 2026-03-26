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
from backend.models.investigation import Hypothesis, TheoryVerdict


# ============================================================
# REQUEST SCHEMAS — what the client sends TO our API
# ============================================================

class InvestigateRequest(BaseModel):
    """
    The request body for POST /investigate.
    """
    mystery: str = Field(
        ...,
        min_length=10,
        max_length=1000,
        description="The mystery or unexplained event to investigate",
        examples=["What happened to the hikers at Dyatlov Pass in 1959?"],
    )


class TestTheoryRequest(BaseModel):
    """
    The request body for POST /test-theory.
    
    Example request body:
    {
        "mystery": "What happened at Dyatlov Pass?",
        "user_theory": "An avalanche forced them out of the tent"
    }
    """
    mystery: str = Field(
        ...,
        min_length=10,
        max_length=1000,
        description="The mystery to investigate",
    )
    user_theory: str = Field(
        ...,
        min_length=5,
        max_length=500,
        description="The user's personal theory to test",
        examples=["An avalanche forced them out of the tent"],
    )


# ============================================================
# RESPONSE SCHEMAS — what our API sends BACK to the client
# ============================================================

class InvestigateResponse(BaseModel):
    """The response body for POST /investigate."""
    mystery: str = Field(..., description="The original mystery that was investigated")
    hypotheses: List[Hypothesis] = Field(
        ...,
        description="Generated hypotheses for the mystery",
    )
    model: str = Field(..., description="LLM model used for generation")


class TestTheoryResponse(BaseModel):
    """The response body for POST /test-theory."""
    mystery: str = Field(..., description="The original mystery")
    user_theory: str = Field(..., description="The user's submitted theory")
    verdict: TheoryVerdict = Field(..., description="The verdict on the user's theory")
    hypotheses: List[Hypothesis] = Field(..., description="All hypotheses tested")
