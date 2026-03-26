"""
shadowmind/backend/agent/theory_pipeline.py

Responsibility: A separate LangGraph StateGraph for Theory Test Mode.
Instead of decompose -> hypothesize, this pipeline uses reframe to convert
the user's theory into a testable hypothesis, then reuses the same
retrieve_evidence and score_and_eliminate nodes, then runs a special verdict.

Run from terminal:
    python -m backend.agent.theory_pipeline "Bermuda Triangle" "Methane gas eruptions caused the disappearances"

Used by: backend/api/main.py (POST /test-theory, WebSocket)
Depends on: backend/agent/nodes.py
"""

import sys
import os

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from langgraph.graph import StateGraph, END

from backend.agent.state import InvestigationState
from backend.agent.nodes import (
    reframe,
    retrieve_evidence,
    score_and_eliminate,
    verdict,
)
from backend.agent.pipeline import retrieve_evidence_with_retry


# ============================================================
# CONDITIONAL EDGE — same logic as normal mode
# ============================================================

def should_continue_or_retry_theory(state: InvestigationState) -> str:
    """Guard after score_and_eliminate in theory mode."""
    hypotheses = state["hypotheses"]
    retry_count = state.get("retry_count", 0)
    survivors = [h for h in hypotheses if h.status == "surviving"]
    
    if len(survivors) > 0:
        print(f"\n  DECISION: {len(survivors)} hypothesis(es) survived -> VERDICT")
        return "verdict"
    elif retry_count < 2:
        print(f"\n  DECISION: 0 survivors, retry {retry_count + 1}/2 -> RETRIEVE again")
        return "retrieve_evidence"
    else:
        print(f"\n  DECISION: 0 survivors after 2 retries -> forcing VERDICT")
        return "force_verdict"


# ============================================================
# BUILD THE THEORY GRAPH
# ============================================================

def build_theory_graph() -> StateGraph:
    """
    Theory Mode graph:
    reframe -> retrieve_evidence -> score_and_eliminate -> verdict
                                        |
                                        +-> retrieve_evidence (retry)
                                        +-> verdict (forced)
    """
    graph = StateGraph(InvestigationState)
    
    # Add nodes
    graph.add_node("reframe", reframe)
    graph.add_node("retrieve_evidence", retrieve_evidence_with_retry)
    graph.add_node("score_and_eliminate", score_and_eliminate)
    graph.add_node("verdict", verdict)
    
    # Set entry point
    graph.set_entry_point("reframe")
    
    # Edges
    graph.add_edge("reframe", "retrieve_evidence")
    graph.add_edge("retrieve_evidence", "score_and_eliminate")
    
    # Conditional edge after scoring
    graph.add_conditional_edges(
        "score_and_eliminate",
        should_continue_or_retry_theory,
        {
            "verdict": "verdict",
            "retrieve_evidence": "retrieve_evidence",
            "force_verdict": "verdict",
        }
    )
    
    graph.add_edge("verdict", END)
    
    return graph


# ============================================================
# PUBLIC API
# ============================================================

def run_theory_pipeline(mystery: str, user_theory: str) -> InvestigationState:
    """Run the theory test pipeline and return the final state."""
    
    print(f"\n{'='*60}")
    print(f"SHADOWMIND THEORY TEST")
    print(f"{'='*60}")
    print(f"Mystery: {mystery}")
    print(f"Theory: {user_theory}")
    
    graph = build_theory_graph()
    app = graph.compile()
    
    initial_state = {
        "mystery": mystery,
        "angles": [],
        "hypotheses": [],
        "evidence": {},
        "scored_hypotheses": [],
        "retry_count": 0,
        "conclusion": None,
        "user_theory": user_theory,
        "theory_mode": True,
        "theory_verdict": None,
    }
    
    final_state = app.invoke(initial_state)
    
    print(f"\n{'='*60}")
    print(f"THEORY TEST COMPLETE")
    print(f"{'='*60}")
    
    return final_state


# ============================================================
# TERMINAL TEST
# ============================================================
if __name__ == "__main__":
    if len(sys.argv) >= 3:
        mystery = sys.argv[1]
        user_theory = " ".join(sys.argv[2:])
    else:
        mystery = "What really happened at the Bermuda Triangle?"
        user_theory = "Methane gas eruptions from the ocean floor caused ships to sink"
    
    final = run_theory_pipeline(mystery, user_theory)
    
    tv = final.get("theory_verdict")
    if tv:
        print(f"\n{'='*60}")
        print(f"FINAL VERDICT")
        print(f"{'='*60}")
        print(f"Verdict: {tv.verdict_label} ({tv.user_theory_confidence:.0%})")
        print(f"Survived: {tv.did_user_theory_survive}")
        if tv.stronger_alternative:
            print(f"Stronger Alternative: {tv.stronger_alternative}")
        print(f"\n{tv.summary}")
