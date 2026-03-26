"""
shadowmind/backend/agent/streaming.py

Responsibility: Wraps the LangGraph pipeline to emit structured JSON events
after each node completes. These events are sent over a WebSocket connection
so the frontend can render the investigation in real-time.

WHY THIS FILE EXISTS:
The normal pipeline.py runs synchronously and prints to terminal.
This wrapper intercepts each node's output and converts it into a 
structured WebSocket event that a browser (or test client) can consume.

Used by: backend/api/main.py (WebSocket endpoint)
Depends on: backend/agent/pipeline.py, backend/agent/nodes.py
"""

import json
from datetime import datetime, timezone
from fastapi import WebSocket

from backend.agent.state import InvestigationState
from backend.agent.pipeline import build_graph
from backend.agent.theory_pipeline import build_theory_graph
from backend.models.investigation import EvidenceChunk


def _serialize_evidence_chunk(chunk: EvidenceChunk) -> dict:
    """
    Converts an EvidenceChunk into a rich JSON dict with domain/favicon metadata.
    
    WHY: The Pydantic model stores domain/favicon inside graph_entities.
    This function unpacks that into the clean format the frontend expects.
    """
    base = chunk.model_dump()
    
    # Extract the domain and favicon from graph_entities (stored as [domain, favicon_url])
    entities = chunk.graph_entities
    if len(entities) >= 2:
        base["source_domain"] = entities[0]
        base["favicon"] = entities[1]
    else:
        # Fallback if metadata wasn't populated
        try:
            domain = chunk.source_url.split("//")[-1].split("/")[0]
        except Exception:
            domain = "unknown"
        base["source_domain"] = domain
        base["favicon"] = f"https://www.google.com/s2/favicons?domain={domain}"
    
    # Rename 'excerpt' to 'text' for the frontend
    base["text"] = base.pop("excerpt", "")
    base["source_title"] = base.get("source_name", "Unknown")
    
    return base


def _make_event(event_type: str, **kwargs) -> dict:
    """Creates a timestamped event dict."""
    return {
        "event": event_type,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **kwargs,
    }


async def stream_investigation(websocket: WebSocket, mystery: str):
    """
    Runs the full LangGraph pipeline and streams structured JSON events
    to the WebSocket after each node completes.
    
    This is the core function called by the /ws/investigate endpoint.
    """
    
    try:
        # Build the graph and compile it
        graph = build_graph()
        app = graph.compile()
        
        # Initial state
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
        
        # Send a starting event
        await websocket.send_json(_make_event(
            "investigation_started",
            mystery=mystery,
        ))
        
        # Stream the graph node by node
        # LangGraph's .stream() yields (node_name, state_update) after each node
        for event in app.stream(initial_state):
            # event is a dict like {"decompose": {"angles": [...]}}
            for node_name, node_output in event.items():
                
                # Build the data payload based on which node just completed
                data = {}
                
                if node_name == "decompose":
                    data["angles"] = node_output.get("angles", [])
                
                elif node_name == "hypothesize":
                    hypotheses = node_output.get("hypotheses", [])
                    data["hypotheses"] = [h.model_dump() for h in hypotheses]
                
                elif node_name == "retrieve_evidence":
                    evidence = node_output.get("evidence", {})
                    # Serialize each evidence chunk with the enriched metadata
                    data["evidence"] = {
                        hyp_id: [_serialize_evidence_chunk(c) for c in chunks]
                        for hyp_id, chunks in evidence.items()
                    }
                
                elif node_name == "score_and_eliminate":
                    scored = node_output.get("scored_hypotheses", [])
                    hypotheses = node_output.get("hypotheses", [])
                    
                    data["scored_hypotheses"] = [sh.model_dump() for sh in scored]
                    data["hypotheses"] = [h.model_dump() for h in hypotheses]
                    
                    # Send individual elimination events for each killed hypothesis
                    for h in hypotheses:
                        if h.status == "eliminated":
                            # Find the corresponding score
                            score = next(
                                (sh.confidence_score for sh in scored if sh.hypothesis_id == h.id),
                                0.0
                            )
                            await websocket.send_json(_make_event(
                                "hypothesis_eliminated",
                                hypothesis_id=h.id,
                                reason=h.elimination_reason or "Unknown",
                                confidence_score=score,
                            ))
                
                elif node_name == "conclude":
                    conclusion = node_output.get("conclusion")
                    if conclusion:
                        data["conclusion"] = conclusion.model_dump()
                
                # Send the node_complete event
                await websocket.send_json(_make_event(
                    "node_complete",
                    node=node_name,
                    data=data,
                ))
        
        # Send the final investigation_complete event
        # We need to get the conclusion from the last state
        # The stream already sent it via conclude node, send a summary event
        await websocket.send_json(_make_event(
            "investigation_complete",
        ))
        
    except Exception as e:
        await websocket.send_json(_make_event(
            "error",
            message=str(e),
        ))


async def stream_theory_test(websocket: WebSocket, mystery: str, user_theory: str):
    """
    Runs the theory test pipeline and streams events to the WebSocket.
    Reuses the same node serialization logic as the normal investigation.
    """
    
    try:
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
        
        await websocket.send_json(_make_event(
            "theory_test_started",
            mystery=mystery,
            user_theory=user_theory,
        ))
        
        for event in app.stream(initial_state):
            for node_name, node_output in event.items():
                data = {}
                
                if node_name == "reframe":
                    hypotheses = node_output.get("hypotheses", [])
                    data["hypotheses"] = [h.model_dump() for h in hypotheses]
                
                elif node_name == "retrieve_evidence":
                    evidence = node_output.get("evidence", {})
                    data["evidence"] = {
                        hyp_id: [_serialize_evidence_chunk(c) for c in chunks]
                        for hyp_id, chunks in evidence.items()
                    }
                
                elif node_name == "score_and_eliminate":
                    scored = node_output.get("scored_hypotheses", [])
                    hypotheses = node_output.get("hypotheses", [])
                    data["scored_hypotheses"] = [sh.model_dump() for sh in scored]
                    data["hypotheses"] = [h.model_dump() for h in hypotheses]
                    
                    for h in hypotheses:
                        if h.status == "eliminated":
                            score = next(
                                (sh.confidence_score for sh in scored if sh.hypothesis_id == h.id),
                                0.0
                            )
                            await websocket.send_json(_make_event(
                                "hypothesis_eliminated",
                                hypothesis_id=h.id,
                                reason=h.elimination_reason or "Unknown",
                                confidence_score=score,
                            ))
                
                elif node_name == "verdict":
                    tv = node_output.get("theory_verdict")
                    if tv:
                        data["theory_verdict"] = tv.model_dump()
                        # Send the special theory_verdict event
                        await websocket.send_json(_make_event(
                            "theory_verdict",
                            did_survive=tv.did_user_theory_survive,
                            verdict_label=tv.verdict_label,
                            user_theory_confidence=tv.user_theory_confidence,
                            summary=tv.summary,
                        ))
                
                await websocket.send_json(_make_event(
                    "node_complete",
                    node=node_name,
                    data=data,
                ))
        
        await websocket.send_json(_make_event("investigation_complete"))
        
    except Exception as e:
        await websocket.send_json(_make_event("error", message=str(e)))

