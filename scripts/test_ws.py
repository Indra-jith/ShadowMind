"""
Test script for the WebSocket streaming endpoint.

Usage:
1. Start the server:  uvicorn backend.api.main:app --reload
2. Run this script:   python scripts/test_ws.py "What happened at the Bermuda Triangle"

This connects to the WebSocket, sends a mystery, and prints
every structured event as it arrives in real-time.
"""

import asyncio
import json
import sys

try:
    import websockets
except ImportError:
    print("Install websockets first: pip install websockets")
    sys.exit(1)


async def test_investigation(mystery: str):
    """Connect to ShadowMind's WebSocket and stream the investigation."""
    
    uri = "ws://127.0.0.1:8000/ws/investigate"
    
    print(f"Connecting to {uri}...")
    
    async with websockets.connect(uri) as ws:
        # Send the mystery
        await ws.send(json.dumps({"mystery": mystery}))
        print(f"Sent mystery: {mystery}")
        print(f"{'='*60}")
        
        # Listen for events until the server closes the connection
        try:
            async for message in ws:
                event = json.loads(message)
                event_type = event.get("event", "unknown")
                timestamp = event.get("timestamp", "")
                
                if event_type == "investigation_started":
                    print(f"\n[{timestamp}] INVESTIGATION STARTED")
                    print(f"  Mystery: {event.get('mystery', '')}")
                
                elif event_type == "node_complete":
                    node = event.get("node", "unknown")
                    data = event.get("data", {})
                    print(f"\n[{timestamp}] NODE COMPLETE: {node}")
                    
                    if node == "decompose":
                        for angle in data.get("angles", []):
                            print(f"  Angle: {angle}")
                    
                    elif node == "hypothesize":
                        for h in data.get("hypotheses", []):
                            print(f"  [{h['id']}] {h['title']} (score: {h['plausibility_score']})")
                    
                    elif node == "retrieve_evidence":
                        evidence = data.get("evidence", {})
                        for hyp_id, chunks in evidence.items():
                            print(f"  Evidence for {hyp_id}:")
                            for c in chunks:
                                domain = c.get("source_domain", "?")
                                favicon = c.get("favicon", "")
                                text = c.get("text", c.get("excerpt", ""))
                                print(f"    [{domain}] {text[:60]}...")
                                print(f"      favicon: {favicon}")
                    
                    elif node == "score_and_eliminate":
                        for h in data.get("hypotheses", []):
                            status = h.get("status", "?")
                            score = h.get("plausibility_score", 0)
                            symbol = "SURVIVED" if status == "surviving" else "ELIMINATED"
                            print(f"  [{h['id']}] {h['title']} -> {symbol} ({score})")
                    
                    elif node == "conclude":
                        conclusion = data.get("conclusion", {})
                        print(f"  Strongest: {conclusion.get('surviving_hypothesis', '?')}")
                        print(f"  Confidence: {conclusion.get('overall_confidence', 0):.0%} ({conclusion.get('confidence_label', '?')})")
                        print(f"  Summary: {conclusion.get('summary', '')}")
                
                elif event_type == "hypothesis_eliminated":
                    print(f"\n  [ELIMINATED] {event.get('hypothesis_id')} (score: {event.get('confidence_score', 0)})")
                    print(f"    Reason: {event.get('reason', '')[:80]}...")
                
                elif event_type == "investigation_complete":
                    print(f"\n{'='*60}")
                    print(f"INVESTIGATION COMPLETE")
                    print(f"{'='*60}")
                
                elif event_type == "error":
                    print(f"\nERROR: {event.get('message', 'Unknown error')}")
                
        except websockets.exceptions.ConnectionClosed:
            print("\nConnection closed by server.")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        mystery = " ".join(sys.argv[1:])
    else:
        mystery = "What really happened at the Bermuda Triangle?"
    
    asyncio.run(test_investigation(mystery))
