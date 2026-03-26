"""
shadowmind/backend/agent/state.py

Responsibility: Defines the "manila folder" that travels between all 5 nodes.
WHY THIS FILE EXISTS: In LangGraph, every node reads from and writes to a single
shared State object. This TypedDict defines exactly what fields that folder contains.

Used by: nodes.py, pipeline.py
Depends on: typing, backend/models/investigation.py
"""

from typing import TypedDict, Optional
from backend.models.investigation import (
    Hypothesis,
    EvidenceChunk,
    ScoredHypothesis,
    InvestigationConclusion,
    TheoryVerdict,
)


class InvestigationState(TypedDict):
    """
    The manila folder that gets passed from node to node.
    
    Every field here represents one piece of ongoing knowledge about the case.
    Nodes READ what they need and WRITE their output back into this state.
    """
    
    # The original question the user typed in
    mystery: str
    
    # Node 1 output: 3-5 specific angles to investigate
    angles: list[str]
    
    # Node 2 output: exactly 4 competing hypotheses
    hypotheses: list[Hypothesis]
    
    # Node 3 output: evidence chunks organized by hypothesis ID
    evidence: dict[str, list[EvidenceChunk]]
    
    # Node 4 output: each hypothesis with a confidence score and reasoning
    scored_hypotheses: list[ScoredHypothesis]
    
    # How many times we have looped back to retrieve_evidence
    retry_count: int
    
    # Node 5 output: the final verdict (None until Node 5 runs)
    conclusion: Optional[InvestigationConclusion]
    
    # --- THEORY MODE FIELDS ---
    # The user's personal theory to test (empty string in normal mode)
    user_theory: str
    
    # Whether this pipeline is running in theory test mode
    theory_mode: bool
    
    # The verdict on the user's theory (None until verdict node runs)
    theory_verdict: Optional[TheoryVerdict]
