"""
shadowmind/backend/models/investigation.py

Responsibility: Defines ALL the Pydantic data models for ShadowMind's
investigation pipeline. These models are the "contracts" between pipeline
nodes — each node produces structured data that the next node depends on.

Used by: backend/agent/ (all 7 pipeline nodes), backend/api/ (responses)
Depends on: pydantic

WHY THIS FILE EXISTS:
Without these models, the LLM returns raw text and every downstream node
would have to guess the shape of the data. Pydantic guarantees:
  1. Every field exists (no missing data)
  2. Every field has the right type (no "banana" where a float should be)
  3. Invalid data is rejected IMMEDIATELY with a clear error message
"""

# ============================================================
# IMPORTS
# ============================================================

# BaseModel — the parent class for all Pydantic models.
# WHY: When you create a class that inherits from BaseModel, Pydantic
# automatically adds validation, serialization (to JSON), and type checking.
# Without it, you'd write all that validation code by hand for every model.
from pydantic import BaseModel, Field

# Optional — marks a field as "this can be None".
# Literal — restricts a field to specific allowed values (like an enum).
# List — marks a field as a list of items.
# WHY: These are Python's built-in type hints. Pydantic reads them to know
# what type each field should be and enforces it at runtime.
from typing import Optional, Literal, List


# ============================================================
# MODEL 1: Hypothesis
# ============================================================
# WHAT: Represents one possible explanation for a mystery.
# WHEN CREATED: Node 2 (hypothesize) generates 4-5 of these.
# USED BY: Nodes 3, 5, 6, 7 — everything downstream needs these.
#
# Example:
#   Hypothesis(
#       id="hyp_001",
#       title="Avalanche Theory",
#       description="An avalanche forced the hikers to flee...",
#       plausibility_score=0.72,
#       status="active"
#   )
# ============================================================

class Hypothesis(BaseModel):
    # id — a unique identifier for this hypothesis.
    # WHY: Nodes 4-6 need to link evidence back to a specific hypothesis.
    # Without an ID, we couldn't say "this evidence supports hypothesis #2".
    # example="hyp_001" shows up in the API docs as a helpful hint.
    id: str = Field(
        ...,  # ... means "this field is REQUIRED, not optional"
        description="Unique identifier for this hypothesis",
        examples=["hyp_001"],
    )

    # title — a short, descriptive name for the hypothesis.
    # WHY: The frontend displays this as a heading in the HypothesisTracker.
    title: str = Field(
        ...,
        description="Short descriptive title",
        examples=["Avalanche Theory"],
    )

    # description — a detailed explanation of what this hypothesis proposes.
    # WHY: Node 3 reads this to generate targeted search queries. The more
    # specific the description, the better the evidence retrieval in Node 4.
    description: str = Field(
        ...,
        description="Detailed explanation of the hypothesis",
    )

    # plausibility_score — how likely this hypothesis is (0.0 to 1.0).
    # WHY: Node 6 uses this to eliminate hypotheses below 0.35 threshold.
    # ge=0.0 means "greater than or equal to 0" — Pydantic rejects -0.5.
    # le=1.0 means "less than or equal to 1" — Pydantic rejects 1.5.
    plausibility_score: float = Field(
        ...,
        ge=0.0,  # Minimum value — prevents negative scores
        le=1.0,  # Maximum value — prevents scores over 100%
        description="Initial plausibility estimate from 0.0 to 1.0",
    )

    # status — tracks whether this hypothesis is still in the running.
    # WHY: Node 6 changes this from "active" to "eliminated" when a
    # hypothesis fails. Node 7 only processes "surviving" hypotheses.
    # Literal["active", "eliminated", "surviving"] means Pydantic will
    # REJECT any value that isn't one of these three exact strings.
    status: Literal["active", "eliminated", "surviving"] = Field(
        default="active",  # Every new hypothesis starts as "active"
        description="Current status in the investigation pipeline",
    )

    # elimination_reason — why this hypothesis was eliminated (if it was).
    # WHY: The frontend shows this to the user so they understand the
    # investigation's reasoning. Optional because active hypotheses haven't
    # been eliminated yet, so this field would be None for them.
    elimination_reason: Optional[str] = Field(
        default=None,
        description="Why this hypothesis was eliminated (if applicable)",
    )


# ============================================================
# MODEL 2: EvidenceRequirement
# ============================================================
# WHAT: Defines what evidence we need to find for each hypothesis.
# WHEN CREATED: Node 3 (plan_evidence) generates one per hypothesis.
# USED BY: Node 4 (retrieve) uses the search_queries to find evidence.
#
# THIS IS THE CORE INNOVATION — instead of searching for what the user
# asked, we search for what would CONFIRM or DENY each hypothesis.
# ============================================================

class EvidenceRequirement(BaseModel):
    # hypothesis_id — links this requirement back to a specific hypothesis.
    # WHY: Node 4 needs to know which hypothesis each search serves.
    hypothesis_id: str = Field(
        ...,
        description="ID of the hypothesis this evidence is for",
    )

    # search_queries — 3 targeted search strings per hypothesis.
    # WHY: Tavily (our search API) takes a query string. We generate 3
    # different queries to maximize coverage — one might find evidence
    # the others miss. Each query is crafted to find confirming OR
    # denying evidence, not just supporting evidence.
    search_queries: List[str] = Field(
        ...,
        min_length=1,  # Must have at least 1 query
        max_length=5,  # Cap at 5 to stay within API rate limits
        description="Targeted search queries to find evidence",
    )

    # what_would_confirm — describes evidence that would support the hypothesis.
    # WHY: Node 5 uses this as a rubric when scoring evidence. It compares
    # each piece of retrieved evidence against this description to determine
    # if it's supporting evidence.
    what_would_confirm: str = Field(
        ...,
        description="Description of evidence that would support this hypothesis",
    )

    # what_would_deny — describes evidence that would disprove the hypothesis.
    # WHY: This is what makes ShadowMind different from regular search.
    # We actively look for CONTRADICTING evidence, not just confirming.
    # Node 5 uses this to score evidence as contradicting.
    what_would_deny: str = Field(
        ...,
        description="Description of evidence that would disprove this hypothesis",
    )


# ============================================================
# MODEL 3: EvidenceChunk
# ============================================================
# WHAT: A single piece of retrieved evidence.
# WHEN CREATED: Node 4 (retrieve) produces these from search results.
# USED BY: Node 5 (scoring), Node 7 (conclusion).
# ============================================================

class EvidenceChunk(BaseModel):
    # id — unique identifier for this evidence piece.
    id: str = Field(..., description="Unique evidence identifier")

    # source_url — where this evidence came from (for citation).
    source_url: str = Field(..., description="URL of the source")

    # source_name — human-readable source name for the frontend.
    source_name: str = Field(
        ...,
        description="Human-readable source name",
        examples=["Wikipedia", "CIA FOIA Reading Room"],
    )

    # excerpt — the actual text of the evidence.
    # WHY: This is what gets embedded in Qdrant and scored against hypotheses.
    excerpt: str = Field(..., description="The actual evidence text")

    # domain_tag — categorizes the evidence into one of 5 domains.
    # WHY: ShadowMind searches across multiple domains to build a
    # well-rounded investigation. The frontend uses this tag to color-code
    # evidence by domain. Literal restricts to exactly these 5 values.
    domain_tag: Literal["science", "history", "government", "news", "geography"] = Field(
        ...,
        description="Domain category of the evidence",
    )

    # relevance_score — how relevant this evidence is to its hypothesis (0-1).
    relevance_score: float = Field(
        ..., ge=0.0, le=1.0,
        description="Relevance to the target hypothesis",
    )

    # hypothesis_id — which hypothesis this evidence was retrieved for.
    hypothesis_id: str = Field(..., description="Target hypothesis ID")

    # graph_entities — named entities extracted for the knowledge graph.
    # WHY: Node 5 / Week 5 uses these to build the NetworkX knowledge graph
    # that links entities across hypotheses (e.g., "Dyatlov" appears in
    # multiple evidence chunks → strong connection).
    graph_entities: List[str] = Field(
        default_factory=list,  # Defaults to empty list if not provided
        description="Named entities for knowledge graph construction",
    )


# ============================================================
# MODEL 4: ScoredHypothesis
# ============================================================
# WHAT: A hypothesis after evidence has been scored against it.
# WHEN CREATED: Node 5 (score_evidence) produces these.
# USED BY: Node 6 (eliminate) and Node 7 (conclude).
# ============================================================

class ScoredHypothesis(BaseModel):
    # hypothesis_id — links back to the original Hypothesis.
    hypothesis_id: str = Field(..., description="ID of the scored hypothesis")

    # confidence_score — the final confidence after evidence evaluation (0-1).
    # WHY: Node 6 checks if this is below 0.35 to eliminate the hypothesis.
    confidence_score: float = Field(
        ..., ge=0.0, le=1.0,
        description="Confidence score after evidence evaluation",
    )

    # supporting_evidence — IDs of evidence chunks that support this hypothesis.
    supporting_evidence: List[str] = Field(
        default_factory=list,
        description="IDs of supporting evidence chunks",
    )

    # contradicting_evidence — IDs of evidence chunks that contradict this.
    contradicting_evidence: List[str] = Field(
        default_factory=list,
        description="IDs of contradicting evidence chunks",
    )

    # reasoning — the LLM's explanation of why this score was given.
    # WHY: Transparency. The user (and we as developers) need to understand
    # why a hypothesis scored high or low. Debugging without this is impossible.
    reasoning: str = Field(..., description="Explanation of the score")


# ============================================================
# MODEL 5: InvestigationConclusion
# ============================================================
# WHAT: The final output of the entire pipeline.
# WHEN CREATED: Node 7 (conclude) produces exactly one of these.
# USED BY: The API returns this to the frontend.
# ============================================================

class InvestigationConclusion(BaseModel):
    # surviving_hypothesis — the ID of the hypothesis that survived elimination.
    surviving_hypothesis: str = Field(
        ...,
        description="ID of the hypothesis that survived evidence testing",
    )

    # overall_confidence — aggregated confidence across all evidence (0-1).
    overall_confidence: float = Field(
        ..., ge=0.0, le=1.0,
        description="Aggregated confidence score",
    )

    # confidence_label — human-readable confidence level.
    # WHY: Users don't think in 0.0-1.0 floats. "High" is more intuitive
    # than 0.78. The frontend displays this label prominently.
    confidence_label: Literal["Low", "Moderate", "High", "Very High"] = Field(
        ...,
        description="Human-readable confidence level",
    )

    # key_evidence — the top 3 most influential evidence chunk IDs.
    # WHY: The conclusion panel shows these as "the evidence that mattered most".
    key_evidence: List[str] = Field(
        ...,
        description="Top 3 most influential evidence chunk IDs",
    )

    # caveats — honest disclaimers about the investigation's limitations.
    # WHY: No AI investigation is perfect. Listing caveats builds trust
    # and shows the system is self-aware about its limitations.
    caveats: List[str] = Field(
        default_factory=list,
        description="Limitations and disclaimers",
    )

    # summary — the full textual summary of the conclusion.
    summary: str = Field(..., description="Full text summary of the investigation")

    # all_sources — every source URL used in the investigation.
    # WHY: Academic integrity + users can verify claims themselves.
    all_sources: List[str] = Field(
        default_factory=list,
        description="All source URLs used in the investigation",
    )


# ============================================================
# MODEL 6: TheoryVerdict
# ============================================================
# WHAT: The output of Theory Test Mode — tells the user if their
#       personal theory was confirmed or destroyed by the evidence.
# WHEN CREATED: The "verdict" node in the theory pipeline.
# USED BY: API response for POST /test-theory, WebSocket events.
# ============================================================

class TheoryVerdict(BaseModel):
    # did_user_theory_survive — the binary answer: did it pass the evidence test?
    did_user_theory_survive: bool = Field(
        ...,
        description="Whether the user's theory survived evidence scoring",
    )

    # user_theory_confidence — the numeric score the LLM judge gave the user's theory
    user_theory_confidence: float = Field(
        ..., ge=0.0, le=1.0,
        description="Confidence score for the user's theory after evidence evaluation",
    )

    # verdict_label — the human-readable verdict
    verdict_label: Literal[
        "CONFIRMED", "PARTIALLY SUPPORTED", "INSUFFICIENT EVIDENCE", "CONTRADICTED"
    ] = Field(
        ...,
        description="Categorical verdict on the user's theory",
    )

    # supporting_evidence — evidence chunk IDs that back the user's theory
    supporting_evidence: List[str] = Field(
        default_factory=list,
        description="Evidence IDs supporting the user's theory",
    )

    # contradicting_evidence — evidence chunk IDs that challenge the user's theory
    contradicting_evidence: List[str] = Field(
        default_factory=list,
        description="Evidence IDs contradicting the user's theory",
    )

    # stronger_alternative — if a different theory scored higher, which one?
    stronger_alternative: Optional[str] = Field(
        default=None,
        description="Title of a stronger alternative theory if one scored higher",
    )

    # summary — the full textual verdict paragraph
    summary: str = Field(..., description="Full text summary of the theory verdict")

