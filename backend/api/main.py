"""
shadowmind/backend/api/main.py

Responsibility: The entry point of our entire backend application.
This file creates the FastAPI app and defines all HTTP endpoints.

Day 1: Basic /investigate endpoint returning raw LLM text.
Day 2: Upgraded to structured outputs — the LLM now returns JSON
       that maps to our Pydantic Hypothesis models.

Used by: uvicorn (the server that runs this app)
Depends on: fastapi, groq SDK, python-dotenv, pydantic, backend/models
"""

# ============================================================
# IMPORTS — every import is explained below
# ============================================================

# os — Python's built-in module for accessing environment variables.
# WHY: We need to read our GROQ_API_KEY from the .env file.
import os

# json — Python's built-in JSON parser.
# WHY: The LLM returns a JSON string. We need to parse it into a
# Python dictionary before Pydantic can validate it.
# What it does under the hood: takes a string like '{"key": "value"}'
# and converts it to a Python dict {"key": "value"}.
import json

# FastAPI — the web framework that handles HTTP requests and responses.
# WHY: gives us routing, validation, auto-docs, and async support.
from fastapi import FastAPI, HTTPException

# Groq — the official Python SDK for the Groq API.
# WHY: handles HTTP calls, authentication, and response parsing for us.
from groq import Groq

# load_dotenv — loads key-value pairs from a .env file into environment variables.
# WHY: we NEVER hardcode API keys in source code.
from dotenv import load_dotenv

# Import our API schemas — WHY: these define the exact shape of
# requests and responses. FastAPI uses them to validate input and
# generate accurate docs at /docs.
from backend.api.schemas import InvestigateRequest, InvestigateResponse

# Import our Hypothesis model — WHY: we parse the LLM's JSON output
# into validated Hypothesis objects. If the LLM returns bad data
# (wrong types, missing fields), Pydantic catches it here.
from backend.models.investigation import Hypothesis

# ============================================================
# SETUP — things that run ONCE when the server starts
# ============================================================

# Load all secrets from the .env file into environment variables — WHY: this
# must happen BEFORE we try to read any env vars, otherwise os.getenv returns None.
load_dotenv()

# Create the FastAPI application instance — WHY: this is the central object that
# holds all our routes. uvicorn looks for this object to know what to serve.
app = FastAPI(
    title="ShadowMind",
    description="What they don't want you to find.",
    version="0.2.0",  # Bumped to 0.2.0 — we now return structured data
)

# Create the Groq client — WHY: we create it ONCE at startup and reuse it
# for every request (connection reuse — explained in Day 1 comprehension check).
groq_client = Groq(
    api_key=os.getenv("GROQ_API_KEY"),
)


# ============================================================
# ROUTES
# ============================================================

@app.get("/")
def health_check():
    """GET / — simple health check."""
    return {
        "project": "ShadowMind",
        "tagline": "What they don't want you to find.",
        "status": "operational",
        "version": "0.2.0",
    }


# response_model=InvestigateResponse tells FastAPI: "the response from this
# endpoint MUST match the InvestigateResponse schema". If it doesn't, FastAPI
# throws a server error instead of sending bad data to the client.
@app.post("/investigate", response_model=InvestigateResponse)
def investigate(request: InvestigateRequest):
    """
    POST /investigate — accepts a mystery and returns structured hypotheses.

    INPUT:  JSON body with "mystery" field (validated by InvestigateRequest)
    OUTPUT: JSON with structured Hypothesis objects (validated by InvestigateResponse)

    Day 1: returned raw LLM text.
    Day 2: returns validated Pydantic Hypothesis objects as JSON.

    Example request body:
      {"mystery": "What happened to the hikers at Dyatlov Pass in 1959?"}
    """

    # Build the system prompt — WHY: this is now MUCH more specific than Day 1.
    # We tell the LLM to return JSON in a precise format that matches our
    # Hypothesis Pydantic model. Without this structure, the LLM would return
    # free-form text that can't be parsed into Hypothesis objects.
    system_prompt = """You are ShadowMind, an autonomous investigation agent.

When given a mystery, generate exactly 4 competing hypotheses.

You MUST respond with ONLY a valid JSON object in this exact format:
{
  "hypotheses": [
    {
      "id": "hyp_001",
      "title": "Short descriptive title",
      "description": "Detailed explanation of what this hypothesis proposes and why it could be true.",
      "plausibility_score": 0.65,
      "status": "active"
    }
  ]
}

RULES:
- Generate exactly 4 hypotheses with IDs hyp_001 through hyp_004
- Each plausibility_score must be between 0.0 and 1.0
- All scores across hypotheses should sum to approximately 1.0
- Status must always be "active" at this stage
- Be creative and consider non-obvious angles
- DO NOT include any text outside the JSON object
- DO NOT use markdown code blocks — return raw JSON only"""

    # Build the user prompt — same as Day 1 but now the LLM knows to
    # return JSON instead of paragraphs.
    user_prompt = f"Investigate this mystery: {request.mystery}"

    # ⚠️ COST WARNING: This call uses Groq API (Llama 3.3 70B).
    # Free tier limit: 30 requests/minute, 6000 tokens/minute.
    # Each call with structured output uses ~600-1000 tokens.

    # Send the prompt to Groq with JSON mode enabled — WHY: response_format
    # tells Groq to force the LLM to output valid JSON. Without this, the
    # LLM might add explanatory text around the JSON ("Here are my hypotheses:
    # {...}") which would break json.loads().
    llm_response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.7,
        max_tokens=2048,  # Increased from 1024 — structured JSON needs more tokens

        # response_format — THIS IS THE KEY CHANGE from Day 1.
        # type: "json_object" tells Groq to constrain the LLM's output to
        # valid JSON. The LLM literally cannot return non-JSON text.
        # WHY: Without this, even with our detailed system prompt, the LLM
        # might occasionally add "Here are the hypotheses:" before the JSON,
        # which would cause json.loads() to crash.
        response_format={"type": "json_object"},
    )

    # Extract the raw JSON string from the LLM response — same as Day 1.
    raw_json = llm_response.choices[0].message.content

    # Parse the JSON string into a Python dictionary — WHY: json.loads()
    # converts the string '{"hypotheses": [...]}' into an actual Python dict
    # that we can access with dict["hypotheses"]. If the JSON is malformed
    # (shouldn't happen with json_object mode, but safety first), we catch
    # the error and return a clear message instead of crashing.
    try:
        parsed = json.loads(raw_json)
    except json.JSONDecodeError as e:
        # HTTPException — a FastAPI helper that returns an HTTP error response.
        # 500 = "Internal Server Error" — something went wrong on our side.
        # WHY: We don't want the user to see a raw Python traceback.
        raise HTTPException(
            status_code=500,
            detail=f"LLM returned invalid JSON: {str(e)}",
        )

    # Validate each hypothesis through Pydantic — THIS IS WHERE THE MAGIC
    # HAPPENS. For each hypothesis dict in the parsed JSON, we create a
    # Hypothesis Pydantic object. If ANY field is missing, has the wrong type,
    # or is out of range (e.g., plausibility_score = 1.5), Pydantic raises
    # a ValidationError with an exact description of what's wrong.
    try:
        hypotheses = [
            Hypothesis(**h)  # ** unpacks the dict into keyword arguments
            for h in parsed.get("hypotheses", [])
            # .get("hypotheses", []) safely gets the list, or empty list
            # if "hypotheses" key is missing (prevents KeyError crash)
        ]
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"LLM output failed Pydantic validation: {str(e)}",
        )

    # Check that we actually got hypotheses — WHY: if the LLM returned
    # valid JSON but with an empty hypotheses list, we should tell the
    # user instead of returning an empty result.
    if not hypotheses:
        raise HTTPException(
            status_code=500,
            detail="LLM did not generate any hypotheses. Try rephrasing your mystery.",
        )

    # Return the structured response — WHY: InvestigateResponse guarantees
    # the response has exactly the fields the frontend expects:
    # mystery (str), hypotheses (List[Hypothesis]), model (str).
    return InvestigateResponse(
        mystery=request.mystery,
        hypotheses=hypotheses,
        model="llama-3.3-70b-versatile",
    )
