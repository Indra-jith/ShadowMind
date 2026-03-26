"""
shadowmind/backend/agent/nodes.py

Responsibility: The 5 node functions that form the LangGraph agent pipeline.
Each function receives the InvestigationState (manila folder), reads what it needs,
does its job, and writes its output back into the state.

Node 1: decompose        — Breaks the mystery into 3-5 investigable angles
Node 2: hypothesize      — Generates 4 competing hypotheses
Node 3: retrieve_evidence — Searches Qdrant for evidence PER HYPOTHESIS
Node 4: score_and_eliminate — LLM scores hypotheses against evidence, kills weak ones
Node 5: conclude         — Writes the final verdict

Used by: pipeline.py
Depends on: groq, backend/rag/database.py, backend/models/investigation.py
"""

# ============================================================
# IMPORTS
# ============================================================

import os
import json
import uuid
from dotenv import load_dotenv
from groq import Groq
from tavily import TavilyClient

# Load environment variables
env_path = os.path.join(os.path.dirname(__file__), '..', '..', '.env')
load_dotenv(dotenv_path=env_path)

# Import the shared state definition
from backend.agent.state import InvestigationState

# Import our existing Pydantic models — REUSED, not rewritten
from backend.models.investigation import (
    Hypothesis,
    EvidenceChunk,
    ScoredHypothesis,
    InvestigationConclusion,
    TheoryVerdict,
)

# Import our existing Qdrant search function — REUSED, not rewritten
from backend.rag.database import search_evidence

# Create the Groq client ONCE (connection reuse)
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Create the Tavily client for live web search fallback
# WHY: When Qdrant doesn't have strong enough evidence (relevance < 0.55),
# we fall back to a live web search to fill the gap.
tavily_key = os.getenv("TAVILY_API_KEY")
tavily_client = TavilyClient(api_key=tavily_key) if tavily_key else None

# The minimum Qdrant relevance score before we trigger a Tavily fallback
QDRANT_QUALITY_THRESHOLD = 0.55


def _extract_url_metadata(url: str) -> dict:
    """
    Given a URL like 'https://en.wikipedia.org/wiki/Bermuda_Triangle',
    extract the domain and build a favicon URL.
    
    Returns: {"source_domain": "en.wikipedia.org", "favicon": "https://..."}
    """
    try:
        # Split the URL to get the domain part
        domain = url.split("//")[-1].split("/")[0]
    except Exception:
        domain = "unknown"
    
    return {
        "source_domain": domain,
        "favicon": f"https://www.google.com/s2/favicons?domain={domain}",
    }


# ============================================================
# HELPER: Call Groq and get JSON back
# ============================================================

def _call_groq_json(system_prompt: str, user_prompt: str) -> dict:
    """
    Sends a prompt to Groq and forces it to return valid JSON.
    
    WHY: Every node needs to call Groq and parse the JSON response.
    Instead of repeating the same 15 lines of code in every node,
    we extract it into one shared helper function (DRY principle).
    """
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.7,
        max_tokens=4096,
        response_format={"type": "json_object"},
    )
    
    raw = response.choices[0].message.content
    return json.loads(raw)


# ============================================================
# NODE 1: DECOMPOSE
# ============================================================
# INPUT:  state["mystery"] → "What happened at the Bermuda Triangle?"
# OUTPUT: state["angles"]  → ["Ship disappearances", "Aircraft incidents", "Ocean conditions", ...]
#
# WHY THIS NODE EXISTS:
# If you ask an LLM "What happened at the Bermuda Triangle?" it gives you a
# surface-level generic answer. But if you first DECOMPOSE it into specific
# angles like "What ships disappeared?" and "What were the weather conditions?",
# the downstream hypotheses become much more targeted and evidence-grounded.
# ============================================================

def decompose(state: InvestigationState) -> dict:
    """Node 1: Break a mystery into 3-5 specific investigable angles."""
    
    mystery = state["mystery"]
    print(f"\n{'='*60}")
    print(f"🔍 NODE 1: DECOMPOSE")
    print(f"{'='*60}")
    print(f"Mystery: {mystery}")
    
    system_prompt = """You are ShadowMind's Decomposition Engine.
Given a mystery, break it down into 3-5 specific, investigable angles.
Each angle should be a focused sub-question that can be independently researched.

Respond with ONLY valid JSON:
{
    "angles": ["angle 1", "angle 2", "angle 3"]
}

RULES:
- Each angle must be specific and searchable (not vague)
- 3-5 angles maximum
- DO NOT include any text outside the JSON"""

    user_prompt = f"Decompose this mystery into investigable angles: {mystery}"
    
    parsed = _call_groq_json(system_prompt, user_prompt)
    angles = parsed.get("angles", [])
    
    # Print the trace so the user can watch the detective think
    for i, angle in enumerate(angles):
        print(f"  📐 Angle {i+1}: {angle}")
    
    # Write the angles back into the manila folder
    return {"angles": angles}


# ============================================================
# NODE 2: HYPOTHESIZE
# ============================================================
# INPUT:  state["mystery"] + state["angles"]
# OUTPUT: state["hypotheses"] → [Hypothesis(...), Hypothesis(...), ...]
#
# WHY THIS NODE EXISTS:
# The angles tell us WHERE to look. But we also need competing THEORIES
# about what happened. Hypotheses are the backbone of the investigation.
# Without them, we are just collecting random facts with no direction.
# ============================================================

def hypothesize(state: InvestigationState) -> dict:
    """Node 2: Generate exactly 4 competing hypotheses based on the angles."""
    
    mystery = state["mystery"]
    angles = state["angles"]
    
    print(f"\n{'='*60}")
    print(f"🧠 NODE 2: HYPOTHESIZE")
    print(f"{'='*60}")
    
    # Format the angles into a readable list for the LLM
    angles_text = "\n".join(f"- {a}" for a in angles)
    
    system_prompt = """You are ShadowMind's Hypothesis Generator.
Given a mystery and its investigable angles, generate exactly 4 competing hypotheses.

Respond with ONLY valid JSON:
{
    "hypotheses": [
        {
            "id": "hyp_001",
            "title": "Short title",
            "description": "Detailed explanation of what this hypothesis proposes.",
            "plausibility_score": 0.65,
            "status": "active"
        }
    ]
}

RULES:
- Generate exactly 4 hypotheses (hyp_001 through hyp_004)
- Each plausibility_score between 0.0 and 1.0
- All scores should sum to approximately 1.0
- Status must be "active"
- Be creative and consider non-obvious angles
- DO NOT include any text outside the JSON"""

    user_prompt = f"""Mystery: {mystery}

Investigable Angles:
{angles_text}

Generate 4 competing hypotheses."""
    
    parsed = _call_groq_json(system_prompt, user_prompt)
    
    # Validate each hypothesis through our existing Pydantic model
    hypotheses = [Hypothesis(**h) for h in parsed.get("hypotheses", [])]
    
    # Print the trace
    for h in hypotheses:
        print(f"  💡 [{h.id}] {h.title} (plausibility: {h.plausibility_score})")
    
    return {"hypotheses": hypotheses}


# ============================================================
# NODE 3: RETRIEVE EVIDENCE
# ============================================================
# INPUT:  state["hypotheses"]
# OUTPUT: state["evidence"] → {"hyp_001": [EvidenceChunk, ...], ...}
#
# WHY THIS NODE EXISTS:
# This is the CORE INNOVATION of ShadowMind.
#
# QUERY-DRIVEN RETRIEVAL (what most RAG apps do):
#   User asks "What happened at Bermuda Triangle?"
#   → System searches Qdrant for "Bermuda Triangle"
#   → Returns generic facts about the topic
#
# HYPOTHESIS-DRIVEN RETRIEVAL (what ShadowMind does):
#   Hypothesis says "Methane gas eruptions caused ships to sink"
#   → System searches Qdrant for "methane gas ocean ship sinking"
#   → Returns TARGETED evidence about that specific theory
#
# The difference: instead of finding random facts, we find evidence
# that specifically SUPPORTS or CONTRADICTS each hypothesis.
# ============================================================

def retrieve_evidence(state: InvestigationState) -> dict:
    """Node 3: Search Qdrant for 3 evidence chunks per hypothesis.
    If the best Qdrant score is below 0.55, call Tavily live web search as fallback."""
    
    hypotheses = state["hypotheses"]
    retry_count = state.get("retry_count", 0)
    
    print(f"\n{'='*60}")
    print(f"NODE 3: RETRIEVE EVIDENCE (Attempt {retry_count + 1})")
    print(f"{'='*60}")
    
    evidence_map = {}
    
    for h in hypotheses:
        # Only search for hypotheses that are still active
        if h.status != "active":
            continue
            
        # Build the search query FROM the hypothesis — this is the innovation
        # If retrying, we broaden the query by adding the mystery itself
        if retry_count > 0:
            search_query = f"{state['mystery']} {h.title} {h.description}"
        else:
            search_query = f"{h.title}: {h.description}"
        
        print(f"\n  Searching for [{h.id}]: {h.title}")
        print(f"     Query: {search_query[:80]}...")
        
        # STEP 1: Search Qdrant first (our local database)
        try:
            results = search_evidence(search_query, limit=3)
        except Exception as e:
            print(f"     Warning: Qdrant search failed: {e}")
            results = []
        
        # Convert Qdrant results into our existing EvidenceChunk Pydantic model
        chunks = []
        for i, point in enumerate(results):
            payload = point.payload
            src_url = payload.get("source_url", "unknown")
            url_meta = _extract_url_metadata(src_url)
            chunk = EvidenceChunk(
                id=f"ev_{h.id}_{i+1}",
                source_url=src_url,
                source_name=payload.get("source_name", "Unknown"),
                excerpt=payload.get("text", ""),
                domain_tag="history",  # Default tag for scraped web content
                relevance_score=round(point.score, 3),
                hypothesis_id=h.id,
                # Store domain metadata in graph_entities for the WebSocket layer
                graph_entities=[url_meta["source_domain"], url_meta["favicon"]],
            )
            chunks.append(chunk)
            print(f"     [Qdrant] Evidence {i+1}: (relevance: {chunk.relevance_score}) {chunk.excerpt[:60]}...")
        
        # STEP 2: Check if Qdrant evidence is strong enough
        best_score = max((c.relevance_score for c in chunks), default=0.0)
        
        if best_score < QDRANT_QUALITY_THRESHOLD and tavily_client:
            # Qdrant evidence is too weak — call Tavily for live web results
            print(f"     Qdrant evidence weak for [{h.id}] (best: {best_score:.3f}) -- calling Tavily live search")
            
            try:
                tavily_results = tavily_client.search(
                    query=search_query,
                    max_results=3,
                    search_depth="basic",
                )
                
                # Convert Tavily results into EvidenceChunk objects
                for j, result in enumerate(tavily_results.get("results", [])):
                    tavily_url = result.get("url", "unknown")
                    tavily_meta = _extract_url_metadata(tavily_url)
                    tavily_chunk = EvidenceChunk(
                        id=f"ev_{h.id}_tavily_{j+1}",
                        source_url=tavily_url,
                        source_name=f"live_web: {tavily_meta['source_domain']}",
                        excerpt=result.get("content", "")[:500],
                        domain_tag="news",  # Live web results tagged as news
                        relevance_score=round(result.get("score", 0.5), 3),
                        hypothesis_id=h.id,
                        graph_entities=[tavily_meta["source_domain"], tavily_meta["favicon"]],
                    )
                    chunks.append(tavily_chunk)
                    print(f"     [Tavily] Evidence {j+1}: {tavily_chunk.excerpt[:60]}...")
                    
            except Exception as e:
                print(f"     Warning: Tavily search failed: {e}")
        elif best_score >= QDRANT_QUALITY_THRESHOLD:
            print(f"     Qdrant evidence strong (best: {best_score:.3f}) -- skipping Tavily")
        elif not tavily_client:
            print(f"     Qdrant evidence weak but TAVILY_API_KEY not set -- skipping live search")
        
        evidence_map[h.id] = chunks
    
    return {"evidence": evidence_map}


# ============================================================
# NODE 4: SCORE AND ELIMINATE
# ============================================================
# INPUT:  state["hypotheses"] + state["evidence"]
# OUTPUT: state["scored_hypotheses"] → [ScoredHypothesis, ...]
#         state["hypotheses"] updated with status/elimination_reason
#
# WHY THIS NODE EXISTS:
# This is where the detective ARGUES WITH ITSELF.
# The LLM reads the evidence and must justify whether each hypothesis
# is supported. If the evidence contradicts the hypothesis, the LLM
# eliminates it with a written reason. This prevents hallucination —
# only evidence-backed theories survive.
# ============================================================

def score_and_eliminate(state: InvestigationState) -> dict:
    """Node 4: Score each hypothesis against its evidence, eliminate weak ones."""
    
    hypotheses = state["hypotheses"]
    evidence = state["evidence"]
    
    print(f"\n{'='*60}")
    print(f"⚖️  NODE 4: SCORE AND ELIMINATE")
    print(f"{'='*60}")
    
    scored = []
    updated_hypotheses = []
    
    for h in hypotheses:
        # Skip already eliminated hypotheses
        if h.status == "eliminated":
            updated_hypotheses.append(h)
            continue
        
        # Gather the evidence text for this hypothesis
        h_evidence = evidence.get(h.id, [])
        if not h_evidence:
            evidence_text = "No evidence was found for this hypothesis."
        else:
            evidence_text = "\n\n".join(
                f"[Evidence {i+1} | Source: {e.source_name} | Relevance: {e.relevance_score}]\n{e.excerpt}"
                for i, e in enumerate(h_evidence)
            )
        
        system_prompt = """You are ShadowMind's Evidence Judge.
You will be given a hypothesis and retrieved evidence. 
Score the hypothesis based on how well the evidence supports or contradicts it.

Respond with ONLY valid JSON:
{
    "confidence_score": 0.65,
    "supporting_evidence_ids": ["ev_hyp_001_1"],
    "contradicting_evidence_ids": ["ev_hyp_001_3"],
    "reasoning": "The evidence shows..."
}

RULES:
- confidence_score between 0.0 and 1.0
- Be HONEST. If evidence contradicts the hypothesis, score it LOW.
- reasoning must explain WHY you gave that score
- DO NOT include any text outside the JSON"""

        user_prompt = f"""Hypothesis [{h.id}]: {h.title}
Description: {h.description}

Retrieved Evidence:
{evidence_text}

Score this hypothesis based on the evidence."""
        
        parsed = _call_groq_json(system_prompt, user_prompt)
        
        confidence = parsed.get("confidence_score", 0.5)
        reasoning = parsed.get("reasoning", "No reasoning provided.")
        
        # Create the ScoredHypothesis using our existing Pydantic model
        sh = ScoredHypothesis(
            hypothesis_id=h.id,
            confidence_score=confidence,
            supporting_evidence=parsed.get("supporting_evidence_ids", []),
            contradicting_evidence=parsed.get("contradicting_evidence_ids", []),
            reasoning=reasoning,
        )
        scored.append(sh)
        
        # ELIMINATION LOGIC: If confidence is below 0.35, kill the hypothesis
        if confidence < 0.35:
            elimination_reason = f"Confidence {confidence:.2f} < 0.35 threshold. {reasoning}"
            h_updated = h.model_copy(update={
                "status": "eliminated",
                "elimination_reason": elimination_reason,
                "plausibility_score": confidence,
            })
            print(f"  ❌ [{h.id}] {h.title} → ELIMINATED (score: {confidence:.2f})")
            print(f"     Reason: {elimination_reason[:80]}...")
        else:
            h_updated = h.model_copy(update={
                "status": "surviving",
                "plausibility_score": confidence,
            })
            print(f"  ✅ [{h.id}] {h.title} → SURVIVED (score: {confidence:.2f})")
        
        updated_hypotheses.append(h_updated)
    
    return {
        "scored_hypotheses": scored,
        "hypotheses": updated_hypotheses,
    }


# ============================================================
# NODE 5: CONCLUDE
# ============================================================
# INPUT:  state["hypotheses"] (survivors) + state["evidence"]
# OUTPUT: state["conclusion"] → InvestigationConclusion(...)
#
# WHY THIS NODE EXISTS:
# The user doesn't want to read raw scores and evidence chunks.
# This node synthesizes everything into a human-readable final report
# with a confidence label, key evidence, and honest caveats.
# ============================================================

def conclude(state: InvestigationState) -> dict:
    """Node 5: Write the final investigation conclusion."""
    
    hypotheses = state["hypotheses"]
    evidence = state["evidence"]
    scored = state["scored_hypotheses"]
    
    print(f"\n{'='*60}")
    print(f"📋 NODE 5: CONCLUDE")
    print(f"{'='*60}")
    
    # Gather all surviving hypotheses
    survivors = [h for h in hypotheses if h.status == "surviving"]
    
    # If nothing survived (forced conclusion after max retries), use all of them
    if not survivors:
        survivors = [h for h in hypotheses if h.status != "eliminated"]
    if not survivors:
        survivors = hypotheses  # Absolute fallback
    
    # Build the evidence text for the conclusion
    all_evidence_text = ""
    all_sources = set()
    for h in survivors:
        h_evidence = evidence.get(h.id, [])
        for e in h_evidence:
            all_evidence_text += f"\n[{e.source_name}]: {e.excerpt}\n"
            all_sources.add(e.source_url)
    
    # Build the hypotheses summary
    hyp_summary = "\n".join(
        f"- [{h.id}] {h.title} (confidence: {h.plausibility_score:.2f}): {h.description}"
        for h in survivors
    )
    
    system_prompt = """You are ShadowMind's Conclusion Writer.
Given surviving hypotheses and supporting evidence, write a final investigation conclusion.

Respond with ONLY valid JSON:
{
    "surviving_hypothesis": "hyp_001",
    "overall_confidence": 0.72,
    "confidence_label": "High",
    "key_evidence": ["ev_hyp_001_1", "ev_hyp_001_2", "ev_hyp_002_1"],
    "caveats": ["Limited primary sources", "..."],
    "summary": "Based on the evidence, the most likely explanation is..."
}

RULES:
- surviving_hypothesis: the ID of the STRONGEST hypothesis
- overall_confidence: 0.0 to 1.0
- confidence_label: "Low" (<0.35), "Moderate" (0.35-0.6), "High" (0.6-0.85), "Very High" (>0.85)
- key_evidence: list the 3 most important evidence IDs
- caveats: at least 2 honest limitations
- summary: a 3-5 sentence conclusion paragraph
- DO NOT include any text outside the JSON"""

    user_prompt = f"""Mystery: {state['mystery']}

Surviving Hypotheses:
{hyp_summary}

Evidence Used:
{all_evidence_text}

Write the final conclusion."""
    
    parsed = _call_groq_json(system_prompt, user_prompt)
    
    # Build the InvestigationConclusion using our existing Pydantic model
    conclusion = InvestigationConclusion(
        surviving_hypothesis=parsed.get("surviving_hypothesis", survivors[0].id),
        overall_confidence=parsed.get("overall_confidence", 0.5),
        confidence_label=parsed.get("confidence_label", "Moderate"),
        key_evidence=parsed.get("key_evidence", []),
        caveats=parsed.get("caveats", []),
        summary=parsed.get("summary", "Investigation concluded."),
        all_sources=list(all_sources),
    )
    
    # Print the final report
    print(f"\n  🏆 Strongest Theory: {conclusion.surviving_hypothesis}")
    print(f"  📊 Confidence: {conclusion.overall_confidence:.2f} ({conclusion.confidence_label})")
    print(f"  📝 Summary: {conclusion.summary}")
    print(f"\n  ⚠️  Caveats:")
    for c in conclusion.caveats:
        print(f"     - {c}")
    
    return {"conclusion": conclusion}


# ============================================================
# THEORY MODE — NODE T1: REFRAME
# ============================================================
# INPUT:  state["mystery"] + state["user_theory"]
# OUTPUT: state["hypotheses"] → [user_theory_as_hypothesis, alt_1, alt_2, alt_3]
#
# WHY THIS NODE EXISTS:
# The user typed a raw theory like "I think the CIA did it".
# We need to convert that into a Hypothesis object (same format as
# normal mode) so the rest of the pipeline can treat it identically.
# We also generate 3 competing theories so the user's theory isn't
# tested in a vacuum — it has to beat alternatives.
# ============================================================

def reframe(state: InvestigationState) -> dict:
    """Theory Mode Node 1: Convert user theory into a testable hypothesis + 3 challengers."""
    
    mystery = state["mystery"]
    user_theory = state["user_theory"]
    
    print(f"\n{'='*60}")
    print(f"THEORY TEST MODE -- Testing: {user_theory}")
    print(f"{'='*60}")
    print(f"Mystery: {mystery}")
    
    system_prompt = """You are ShadowMind's Theory Reframing Engine.
The user has submitted a personal theory about a mystery. Your job:
1. Restate the user's theory as a formal, testable hypothesis (hyp_001)
2. Generate 3 competing alternative hypotheses (hyp_002 through hyp_004)

Respond with ONLY valid JSON:
{
    "hypotheses": [
        {
            "id": "hyp_001",
            "title": "User's restated theory",
            "description": "Formal version of what the user proposed.",
            "plausibility_score": 0.5,
            "status": "active"
        },
        {
            "id": "hyp_002",
            "title": "Alternative 1",
            "description": "A competing explanation.",
            "plausibility_score": 0.3,
            "status": "active"
        }
    ]
}

RULES:
- hyp_001 MUST be the user's theory restated formally
- hyp_002 through hyp_004 are your competing alternatives
- All plausibility scores between 0.0 and 1.0
- The user's theory gets NO advantage in scoring
- Status must be "active"
- DO NOT include any text outside the JSON"""

    user_prompt = f"""Mystery: {mystery}
User's Theory: {user_theory}

Reframe the user's theory and generate 3 competing alternatives."""
    
    parsed = _call_groq_json(system_prompt, user_prompt)
    hypotheses = [Hypothesis(**h) for h in parsed.get("hypotheses", [])]
    
    # Print the trace
    for h in hypotheses:
        label = "(USER THEORY)" if h.id == "hyp_001" else "(CHALLENGER)"
        print(f"  {label} [{h.id}] {h.title} (score: {h.plausibility_score})")
    
    return {"hypotheses": hypotheses}


# ============================================================
# THEORY MODE — NODE T4: VERDICT
# ============================================================
# INPUT:  state["hypotheses"] (scored) + state["evidence"] + state["scored_hypotheses"]
# OUTPUT: state["theory_verdict"] → TheoryVerdict(...)
#
# WHY THIS NODE EXISTS:
# In normal mode, we just pick the strongest survivor.
# In theory mode, the user specifically wants to know: did MY theory
# survive? Was it confirmed? What contradicted it? Was something else
# stronger? This node answers all of those questions directly.
# ============================================================

def verdict(state: InvestigationState) -> dict:
    """Theory Mode Node 4: Deliver the verdict on the user's theory."""
    
    hypotheses = state["hypotheses"]
    evidence = state["evidence"]
    scored = state["scored_hypotheses"]
    
    print(f"\n{'='*60}")
    print(f"VERDICT")
    print(f"{'='*60}")
    
    # Find the user's theory (always hyp_001)
    user_hyp = next((h for h in hypotheses if h.id == "hyp_001"), None)
    user_score_obj = next((s for s in scored if s.hypothesis_id == "hyp_001"), None)
    
    # Check if the user's theory survived
    did_survive = user_hyp is not None and user_hyp.status == "surviving"
    user_confidence = user_score_obj.confidence_score if user_score_obj else 0.0
    
    # Determine the verdict label
    if user_confidence >= 0.7:
        verdict_label = "CONFIRMED"
    elif user_confidence >= 0.45:
        verdict_label = "PARTIALLY SUPPORTED"
    elif user_confidence >= 0.35:
        verdict_label = "INSUFFICIENT EVIDENCE"
    else:
        verdict_label = "CONTRADICTED"
    
    # Find if a stronger alternative exists
    survivors = [h for h in hypotheses if h.status == "surviving" and h.id != "hyp_001"]
    stronger_alt = None
    if survivors:
        best_alt = max(survivors, key=lambda h: h.plausibility_score)
        if best_alt.plausibility_score > user_confidence:
            stronger_alt = best_alt.title
    
    # Gather supporting and contradicting evidence IDs
    supporting = user_score_obj.supporting_evidence if user_score_obj else []
    contradicting = user_score_obj.contradicting_evidence if user_score_obj else []
    
    # Build evidence context for the summary
    user_evidence = evidence.get("hyp_001", [])
    evidence_text = "\n".join(f"[{e.source_name}]: {e.excerpt[:200]}" for e in user_evidence)
    
    system_prompt = """You are ShadowMind's Theory Verdict Writer.
The user submitted a personal theory and it has been tested against evidence.
Write a 3-5 sentence verdict explaining whether the theory held up.

Respond with ONLY valid JSON:
{
    "summary": "Based on the evidence..."
}

RULES:
- Be HONEST. If the theory failed, say so clearly.
- Reference specific evidence that supported or contradicted it
- If an alternative theory was stronger, mention it
- DO NOT include any text outside the JSON"""

    user_prompt = f"""User's Theory: {user_hyp.title if user_hyp else state['user_theory']}
Confidence Score: {user_confidence:.2f}
Verdict: {verdict_label}
Did Survive: {did_survive}
Stronger Alternative: {stronger_alt or 'None'}

Evidence reviewed:
{evidence_text}

Write the verdict summary."""
    
    parsed = _call_groq_json(system_prompt, user_prompt)
    
    theory_verdict = TheoryVerdict(
        did_user_theory_survive=did_survive,
        user_theory_confidence=user_confidence,
        verdict_label=verdict_label,
        supporting_evidence=supporting,
        contradicting_evidence=contradicting,
        stronger_alternative=stronger_alt,
        summary=parsed.get("summary", "Verdict could not be generated."),
    )
    
    # Print the verdict beautifully
    symbol = "CONFIRMED" if did_survive else "REJECTED"
    print(f"\n  VERDICT: {verdict_label} ({user_confidence:.0%})")
    print(f"  User Theory: {symbol}")
    if stronger_alt:
        print(f"  Stronger Alternative: {stronger_alt}")
    print(f"  Summary: {theory_verdict.summary}")
    
    return {"theory_verdict": theory_verdict}

