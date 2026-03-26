"""
shadowmind/backend/agent/pipeline.py

Responsibility: Wires all 5 nodes into a LangGraph StateGraph with conditional edges.
This is the conductor that decides the order of operations and handles retry loops.

Run from terminal:
    python -m backend.agent.pipeline "What happened at the Bermuda Triangle"

Used by: backend/api/main.py (later), terminal testing (now)
Depends on: langgraph, backend/agent/nodes.py, backend/agent/state.py
"""

import sys
import os

# Add project root to path so imports work when running as __main__
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from langgraph.graph import StateGraph, END

# Import our state definition (the manila folder)
from backend.agent.state import InvestigationState

# Import all 5 node functions
from backend.agent.nodes import (
    decompose,
    hypothesize,
    retrieve_evidence,
    score_and_eliminate,
    conclude,
)


# ============================================================
# CONDITIONAL EDGE LOGIC
# ============================================================

def should_continue_or_retry(state: InvestigationState) -> str:
    """
    The guard standing at the door after Node 4 (score_and_eliminate).
    
    Reads the manila folder and decides which door to open:
    - "conclude"           → at least 1 hypothesis survived → proceed to verdict
    - "retrieve_evidence"  → all eliminated + retries left → loop back with broader search
    - "force_conclude"     → all eliminated + max retries hit → force a conclusion anyway
    """
    
    hypotheses = state["hypotheses"]
    retry_count = state.get("retry_count", 0)
    
    # Count how many hypotheses survived
    survivors = [h for h in hypotheses if h.status == "surviving"]
    
    if len(survivors) > 0:
        # At least one hypothesis has evidence backing it — proceed to conclusion
        print(f"\n  ➡️  DECISION: {len(survivors)} hypothesis(es) survived → proceeding to CONCLUDE")
        return "conclude"
    elif retry_count < 2:
        # All eliminated, but we still have retries — loop back with broader queries
        print(f"\n  🔄 DECISION: 0 survivors, retry {retry_count + 1}/2 → looping back to RETRIEVE")
        return "retrieve_evidence"
    else:
        # All eliminated and we've exhausted retries — force a conclusion
        print(f"\n  ⚠️  DECISION: 0 survivors after 2 retries → forcing CONCLUDE")
        return "force_conclude"


# ============================================================
# BUILD THE GRAPH
# ============================================================

def build_graph() -> StateGraph:
    """
    Creates the LangGraph StateGraph with all 5 nodes and edges.
    
    The graph flows like this:
    
    START → decompose → hypothesize → retrieve_evidence → score_and_eliminate
                                                              │
                                                              ├─→ conclude → END
                                                              ├─→ retrieve_evidence (retry)
                                                              └─→ conclude (forced) → END
    """
    
    # Create the graph builder with our state type
    graph = StateGraph(InvestigationState)
    
    # --- ADD NODES (rooms) ---
    graph.add_node("decompose", decompose)
    graph.add_node("hypothesize", hypothesize)
    graph.add_node("retrieve_evidence", retrieve_evidence_with_retry)
    graph.add_node("score_and_eliminate", score_and_eliminate)
    graph.add_node("conclude", conclude)
    
    # --- ADD EDGES (doors between rooms) ---
    
    # The entry point: always start with decompose
    graph.set_entry_point("decompose")
    
    # Normal flow: decompose → hypothesize → retrieve → score
    graph.add_edge("decompose", "hypothesize")
    graph.add_edge("hypothesize", "retrieve_evidence")
    graph.add_edge("retrieve_evidence", "score_and_eliminate")
    
    # CONDITIONAL EDGE: the guard after score_and_eliminate
    graph.add_conditional_edges(
        "score_and_eliminate",
        should_continue_or_retry,
        {
            "conclude": "conclude",
            "retrieve_evidence": "retrieve_evidence",  # Loop back!
            "force_conclude": "conclude",               # Forced conclusion
        }
    )
    
    # After conclude, the investigation is done
    graph.add_edge("conclude", END)
    
    return graph


def retrieve_evidence_with_retry(state: InvestigationState) -> dict:
    """
    Wrapper around retrieve_evidence that increments the retry counter
    when the node is called from the retry loop.
    
    WHY: LangGraph nodes are pure functions — they don't know if they're
    being called for the first time or as a retry. This wrapper handles
    the retry_count increment so the conditional edge knows when to stop.
    """
    # Get current retry count
    retry_count = state.get("retry_count", 0)
    
    # If evidence already exists (means this is a retry), increment
    if state.get("evidence"):
        retry_count += 1
        print(f"\n  🔄 RETRY #{retry_count}: Broadening search queries...")
        
        # Reset surviving hypotheses back to active for re-evaluation
        updated_hypotheses = []
        for h in state["hypotheses"]:
            if h.status == "eliminated":
                h_reset = h.model_copy(update={
                    "status": "active",
                    "elimination_reason": None,
                })
                updated_hypotheses.append(h_reset)
            else:
                updated_hypotheses.append(h)
        
        # Update state before calling retrieve
        state = {**state, "hypotheses": updated_hypotheses, "retry_count": retry_count}
    
    # Call the actual retrieve_evidence node
    result = retrieve_evidence(state)
    result["retry_count"] = retry_count
    
    return result


# ============================================================
# PUBLIC API
# ============================================================

def run_pipeline(mystery: str) -> InvestigationState:
    """
    The single function you call to run the full investigation.
    
    Takes a mystery string, runs every node, and returns the final
    state containing the complete investigation trace.
    """
    
    print(f"\n{'='*60}")
    print(f"🕵️  SHADOWMIND INVESTIGATION STARTING")
    print(f"{'='*60}")
    print(f"Mystery: {mystery}")
    
    # Build the graph and compile it into a runnable
    graph = build_graph()
    app = graph.compile()
    
    # Create the initial state (empty manila folder with just the mystery)
    initial_state = {
        "mystery": mystery,
        "angles": [],
        "hypotheses": [],
        "evidence": {},
        "scored_hypotheses": [],
        "retry_count": 0,
        "conclusion": None,
        "user_theory": "",
        "theory_mode": False,
        "theory_verdict": None,
    }
    
    # Run the graph — LangGraph handles all the node-to-node flow automatically
    final_state = app.invoke(initial_state)
    
    print(f"\n{'='*60}")
    print(f"🏁 INVESTIGATION COMPLETE")
    print(f"{'='*60}")
    
    return final_state


# ============================================================
# TERMINAL TEST
# ============================================================
if __name__ == "__main__":
    # Read the mystery from the command line arguments
    if len(sys.argv) > 1:
        mystery = " ".join(sys.argv[1:])
    else:
        mystery = "What really happened at the Bermuda Triangle?"
    
    final = run_pipeline(mystery)
    
    # Print the final conclusion beautifully
    conclusion = final.get("conclusion")
    if conclusion:
        print(f"\n{'='*60}")
        print(f"📜 FINAL REPORT")
        print(f"{'='*60}")
        print(f"\n{conclusion.summary}")
        print(f"\nConfidence: {conclusion.overall_confidence:.0%} ({conclusion.confidence_label})")
        print(f"\nCaveats:")
        for c in conclusion.caveats:
            print(f"  - {c}")
        print(f"\nSources Used: {len(conclusion.all_sources)}")
