import { useCallback, useEffect, useRef, useState } from 'react';

/* ============================================================
   TYPE DEFINITIONS
   ============================================================ */

export type PipelineStage =
  | 'idle'
  | 'scanning'
  | 'generating'
  | 'retrieving'
  | 'scoring'
  | 'concluded';

export interface Evidence {
  id: string;
  source_url: string;
  source_name: string;
  text: string;
  domain_tag: string;
  relevance_score: number;
  hypothesis_id: string;
  source_domain?: string;
  favicon?: string;
}

export interface Hypothesis {
  id: string;
  title: string;
  description: string;
  plausibility_score: number;
  status: 'pending' | 'active' | 'eliminated' | 'surviving';
  elimination_reason?: string | null;
  evidence: Evidence[];
  confidence?: number;
}

export interface Conclusion {
  surviving_hypothesis: string;
  overall_confidence: number;
  confidence_label: string;
  key_evidence: string[];
  caveats: string[];
  summary: string;
  all_sources: string[];
}

export interface InvestigationState {
  stage: PipelineStage;
  hypotheses: Hypothesis[];
  terminalLogs: string[];
  verdict: Conclusion | null;
  isConnected: boolean;
  error: string | null;
}

/* ============================================================
   NODE NAME → PIPELINE STAGE MAPPING
   ============================================================ */

const NODE_TO_STAGE: Record<string, PipelineStage> = {
  decompose: 'scanning',
  hypothesize: 'generating',
  retrieve_evidence: 'retrieving',
  score_and_eliminate: 'scoring',
  conclude: 'concluded',
};

/* ============================================================
   MOCK SIMULATION
   ============================================================ */

const MOCK_DELAYS: { node: string; delay: number }[] = [
  { node: 'decompose', delay: 2000 },
  { node: 'hypothesize', delay: 3000 },
  { node: 'retrieve_evidence', delay: 4000 },
  { node: 'score_and_eliminate', delay: 3500 },
  { node: 'conclude', delay: 2500 },
];

const MOCK_HYPOTHESES: Hypothesis[] = [
  {
    id: 'hyp_001',
    title: 'Government Cover-Up',
    description: 'Evidence suggests systematic suppression of information by state intelligence agencies, including classified document redactions and witness intimidation campaigns.',
    plausibility_score: 0.72,
    status: 'active',
    evidence: [],
  },
  {
    id: 'hyp_002',
    title: 'Natural Phenomenon',
    description: 'Environmental and atmospheric conditions created a rare convergence of natural events that explain all observed anomalies without requiring human intervention.',
    plausibility_score: 0.58,
    status: 'active',
    evidence: [],
  },
  {
    id: 'hyp_003',
    title: 'Mass Hysteria Event',
    description: 'Social contagion and media amplification transformed ordinary events into an extraordinary narrative through collective misperception and confirmation bias.',
    plausibility_score: 0.41,
    status: 'active',
    evidence: [],
  },
  {
    id: 'hyp_004',
    title: 'Unknown Technology',
    description: 'Physical evidence and expert testimony point to technology beyond publicly known capabilities, possibly from classified military programs or unknown origin.',
    plausibility_score: 0.35,
    status: 'active',
    evidence: [],
  },
];

const MOCK_EVIDENCE: Evidence[] = [
  {
    id: 'ev_001',
    source_url: 'https://en.wikipedia.org/wiki/example',
    source_name: 'Wikipedia',
    text: 'Declassified documents from 1967 reveal that intelligence agencies actively monitored and suppressed civilian reports...',
    domain_tag: 'government',
    relevance_score: 0.89,
    hypothesis_id: 'hyp_001',
    source_domain: 'wikipedia.org',
  },
  {
    id: 'ev_002',
    source_url: 'https://www.nature.com/articles/example',
    source_name: 'Nature',
    text: 'Atmospheric inversion layers combined with temperature gradients can produce optical phenomena consistent with reported observations...',
    domain_tag: 'science',
    relevance_score: 0.76,
    hypothesis_id: 'hyp_002',
    source_domain: 'nature.com',
  },
  {
    id: 'ev_003',
    source_url: 'https://www.history.com/example',
    source_name: 'History.com',
    text: 'Similar patterns of mass sighting events have been documented throughout history, typically following periods of social anxiety...',
    domain_tag: 'history',
    relevance_score: 0.63,
    hypothesis_id: 'hyp_003',
    source_domain: 'history.com',
  },
  {
    id: 'ev_004',
    source_url: 'https://arxiv.org/abs/example',
    source_name: 'ArXiv',
    text: 'Analysis of radar data shows objects exhibiting flight characteristics inconsistent with known aircraft or atmospheric phenomena...',
    domain_tag: 'science',
    relevance_score: 0.71,
    hypothesis_id: 'hyp_004',
    source_domain: 'arxiv.org',
  },
];

const MOCK_LOGS: Record<string, string[]> = {
  decompose: [
    '> INITIALIZING NEURAL DECOMPOSITION...',
    '> PARSING MYSTERY PARAMETERS...',
    '> SCANNING KNOWLEDGE GRAPH [████████████░░░░] 75%',
    '> IDENTIFIED 4 INVESTIGATION VECTORS',
    '> DECOMPOSITION COMPLETE ✓',
  ],
  hypothesize: [
    '> GENERATING HYPOTHESIS VECTORS...',
    '> APPLYING ADVERSARIAL DIVERSITY FILTER...',
    '> HYPOTHESIS VECTORS GENERATED: 4',
    '> PLAUSIBILITY SCORES ASSIGNED',
    '> HYPOTHESIS GENERATION COMPLETE ✓',
  ],
  retrieve_evidence: [
    '> EVIDENCE RETRIEVAL IN PROGRESS...',
    '> QUERYING QDRANT VECTOR DB [████░░░░░░░░░░░░] 25%',
    '> QDRANT HIT: 12 RELEVANT CHUNKS FOUND',
    '> ACTIVATING TAVILY FALLBACK SEARCH...',
    '> TAVILY RESULTS: 8 ADDITIONAL SOURCES',
    '> TOTAL EVIDENCE RETRIEVED: 20 CHUNKS',
    '> EVIDENCE RETRIEVAL COMPLETE ✓',
  ],
  score_and_eliminate: [
    '> SCORING HYPOTHESES AGAINST EVIDENCE...',
    '> HYP_001: CONFIDENCE 0.72 → SURVIVING',
    '> HYP_002: CONFIDENCE 0.58 → SURVIVING',
    '> HYP_003: CONFIDENCE 0.28 → ✗ ELIMINATED',
    '> HYP_004: CONFIDENCE 0.31 → ✗ ELIMINATED',
    '> SURVIVORS: 2 | ELIMINATED: 2',
    '> SCORING COMPLETE ✓',
  ],
  conclude: [
    '> SYNTHESIZING FINAL VERDICT...',
    '> AGGREGATING SURVIVING EVIDENCE...',
    '> GENERATING CONCLUSION REPORT...',
    '> CONFIDENCE LABEL: HIGH',
    '> CASE STATUS: RESOLVED ✓',
  ],
};

function getMockConclusion(): Conclusion {
  return {
    surviving_hypothesis: 'hyp_001',
    overall_confidence: 0.72,
    confidence_label: 'High',
    key_evidence: ['ev_001', 'ev_002'],
    caveats: [
      'Limited access to classified primary sources',
      'Historical accounts may contain factual inaccuracies',
      'Correlation does not imply causation in pattern analysis',
    ],
    summary:
      'After systematic analysis of all available evidence, the investigation concludes that the Government Cover-Up hypothesis (H-001) presents the strongest evidence-backed explanation. Declassified documents and verified witness testimony provide substantial corroboration, while competing hypotheses lacked sufficient evidentiary support to survive rigorous scoring. Two hypotheses were eliminated for confidence scores below the 0.35 threshold.',
    all_sources: [
      'https://en.wikipedia.org/wiki/example',
      'https://www.nature.com/articles/example',
      'https://www.history.com/example',
      'https://arxiv.org/abs/example',
    ],
  };
}

/* ============================================================
   CIRCULAR BUFFER
   ============================================================ */

class CircularBuffer<T> {
  private items: T[] = [];
  private maxSize: number;

  constructor(maxSize: number) {
    this.maxSize = maxSize;
  }

  push(item: T) {
    this.items.push(item);
    if (this.items.length > this.maxSize) {
      this.items.shift();
    }
  }

  pushMany(newItems: T[]) {
    for (const item of newItems) {
      this.push(item);
    }
  }

  getAll(): T[] {
    return [...this.items];
  }

  clear() {
    this.items = [];
  }
}

/* ============================================================
   THE HOOK
   ============================================================ */

const INITIAL_STATE: InvestigationState = {
  stage: 'idle',
  hypotheses: [],
  terminalLogs: [],
  verdict: null,
  isConnected: false,
  error: null,
};

export function useInvestigation() {
  const [state, setState] = useState<InvestigationState>(INITIAL_STATE);
  const wsRef = useRef<WebSocket | null>(null);
  const logsBuffer = useRef(new CircularBuffer<string>(200));
  const reconnectAttempts = useRef(0);
  const reconnectTimer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);
  const mockTimers = useRef<ReturnType<typeof setTimeout>[]>([]);

  const isMockMode = import.meta.env.VITE_MOCK === 'true';

  /* Push log lines and update state */
  const addLogs = useCallback((lines: string[]) => {
    logsBuffer.current.pushMany(lines);
    setState((prev) => ({
      ...prev,
      terminalLogs: logsBuffer.current.getAll(),
    }));
  }, []);

  /* Handle incoming WebSocket messages */
  const handleMessage = useCallback(
    (event: MessageEvent) => {
      try {
        const data = JSON.parse(event.data as string) as Record<string, unknown>;
        const eventType = data.event as string;

        switch (eventType) {
          case 'investigation_started':
            addLogs([
              `> INVESTIGATION INITIATED: ${data.mystery as string}`,
              '> CONNECTING TO NEURAL NETWORK...',
            ]);
            setState((prev) => ({ ...prev, stage: 'scanning' }));
            break;

          case 'node_complete': {
            const nodeName = data.node as string;
            const nodeData = data.data as Record<string, unknown>;

            addLogs([`> NODE COMPLETE: ${nodeName.toUpperCase()}`]);

            setState((prev) => {
              const newStage = NODE_TO_STAGE[nodeName] ?? prev.stage;
              const updated = { ...prev, stage: newStage };

              if (nodeName === 'hypothesize' && nodeData.hypotheses) {
                updated.hypotheses = (
                  nodeData.hypotheses as Array<Record<string, unknown>>
                ).map((h) => ({
                  id: h.id as string,
                  title: h.title as string,
                  description: h.description as string,
                  plausibility_score: h.plausibility_score as number,
                  status: (h.status as Hypothesis['status']) ?? 'active',
                  elimination_reason: h.elimination_reason as string | null,
                  evidence: [],
                }));
              }

              if (
                nodeName === 'retrieve_evidence' &&
                nodeData.evidence
              ) {
                const evidenceMap = nodeData.evidence as Record<string, Evidence[]>;
                updated.hypotheses = prev.hypotheses.map((h) => ({
                  ...h,
                  evidence: evidenceMap[h.id] ?? h.evidence,
                }));
              }

              if (
                nodeName === 'score_and_eliminate' &&
                nodeData.hypotheses
              ) {
                const scoredHypotheses =
                  nodeData.hypotheses as Array<Record<string, unknown>>;
                const scoredMap = new Map(
                  (
                    (nodeData.scored_hypotheses as Array<Record<string, unknown>>) ??
                    []
                  ).map((sh) => [sh.hypothesis_id as string, sh.confidence_score as number])
                );

                updated.hypotheses = prev.hypotheses.map((h) => {
                  const scored = scoredHypotheses.find(
                    (sh) => sh.id === h.id
                  );
                  return {
                    ...h,
                    status: (scored?.status as Hypothesis['status']) ?? h.status,
                    elimination_reason:
                      (scored?.elimination_reason as string) ??
                      h.elimination_reason,
                    confidence: scoredMap.get(h.id) ?? h.confidence,
                  };
                });
              }

              if (nodeName === 'conclude' && nodeData.conclusion) {
                updated.verdict = nodeData.conclusion as Conclusion;
              }

              return updated;
            });
            break;
          }

          case 'hypothesis_eliminated':
            addLogs([
              `> ✗ HYPOTHESIS ${data.hypothesis_id as string} ELIMINATED`,
              `>   REASON: ${data.reason as string}`,
              `>   CONFIDENCE: ${data.confidence_score as number}`,
            ]);
            break;

          case 'investigation_complete':
            addLogs(['> ═══════════════════════════════', '> INVESTIGATION COMPLETE', '> ═══════════════════════════════']);
            break;

          case 'error':
            addLogs([`> ✗ ERROR: ${data.message as string}`]);
            setState((prev) => ({
              ...prev,
              error: data.message as string,
            }));
            break;
        }
      } catch {
        addLogs(['> ✗ FAILED TO PARSE SERVER MESSAGE']);
      }
    },
    [addLogs]
  );

  /* Connect to WebSocket */
  const connect = useCallback(() => {
    if (isMockMode) return;

    const wsUrl = import.meta.env.VITE_WS_URL || 'localhost:8000';
    const ws = new WebSocket(`ws://${wsUrl}/ws/investigate`);

    ws.onopen = () => {
      setState((prev) => ({ ...prev, isConnected: true, error: null }));
      reconnectAttempts.current = 0;
      addLogs(['> WEBSOCKET CONNECTED']);
    };

    ws.onmessage = handleMessage;

    ws.onclose = () => {
      setState((prev) => ({ ...prev, isConnected: false }));
      addLogs(['> WEBSOCKET DISCONNECTED']);

      // Exponential backoff reconnect
      const delay = Math.min(
        1000 * Math.pow(2, reconnectAttempts.current),
        30000
      );
      reconnectAttempts.current++;
      reconnectTimer.current = setTimeout(connect, delay);
    };

    ws.onerror = () => {
      addLogs(['> ✗ WEBSOCKET ERROR']);
    };

    wsRef.current = ws;
  }, [isMockMode, handleMessage, addLogs]);

  /* Start investigation — real or mock */
  const startInvestigation = useCallback(
    (query: string) => {
      // Reset state
      logsBuffer.current.clear();
      setState({
        ...INITIAL_STATE,
        isConnected: isMockMode || state.isConnected,
        terminalLogs: [],
      });

      if (isMockMode) {
        // ── Mock Mode: simulate the pipeline ──
        addLogs([
          `> INVESTIGATION INITIATED: ${query}`,
          '> [MOCK MODE] SIMULATING PIPELINE...',
          '> CONNECTING TO NEURAL NETWORK...',
        ]);
        setState((prev) => ({ ...prev, stage: 'scanning', isConnected: true }));

        let accumulatedDelay = 500;

        MOCK_DELAYS.forEach(({ node, delay }) => {
          accumulatedDelay += delay;

          const timer = setTimeout(() => {
            const stage = NODE_TO_STAGE[node];
            const logs = MOCK_LOGS[node] ?? [];

            if (stage) {
              setState((prev) => {
                const updated = { ...prev, stage };

                if (node === 'hypothesize') {
                  updated.hypotheses = MOCK_HYPOTHESES.map((h) => ({
                    ...h,
                    status: 'active' as const,
                  }));
                }

                if (node === 'retrieve_evidence') {
                  updated.hypotheses = prev.hypotheses.map((h) => ({
                    ...h,
                    evidence: MOCK_EVIDENCE.filter(
                      (e) => e.hypothesis_id === h.id
                    ),
                  }));
                }

                if (node === 'score_and_eliminate') {
                  updated.hypotheses = prev.hypotheses.map((h) => {
                    if (h.id === 'hyp_003') {
                      return {
                        ...h,
                        status: 'eliminated' as const,
                        confidence: 0.28,
                        elimination_reason:
                          'Insufficient evidence to support mass hysteria as primary explanation',
                      };
                    }
                    if (h.id === 'hyp_004') {
                      return {
                        ...h,
                        status: 'eliminated' as const,
                        confidence: 0.31,
                        elimination_reason:
                          'No verifiable physical evidence of unknown technology',
                      };
                    }
                    return {
                      ...h,
                      status: 'surviving' as const,
                      confidence:
                        h.id === 'hyp_001'
                          ? 0.72
                          : 0.58,
                    };
                  });
                }

                if (node === 'conclude') {
                  updated.verdict = getMockConclusion();
                }

                return updated;
              });
            }

            addLogs(logs);
          }, accumulatedDelay);

          mockTimers.current.push(timer);
        });
      } else {
        // ── Real Mode: send via WebSocket ──
        if (wsRef.current?.readyState === WebSocket.OPEN) {
          wsRef.current.send(JSON.stringify({ mystery: query }));
        } else {
          setState((prev) => ({
            ...prev,
            error: 'WebSocket not connected',
          }));
        }
      }
    },
    [isMockMode, state.isConnected, addLogs]
  );

  /* Reset */
  const resetInvestigation = useCallback(() => {
    logsBuffer.current.clear();
    mockTimers.current.forEach(clearTimeout);
    mockTimers.current = [];
    setState({
      ...INITIAL_STATE,
      isConnected: isMockMode || (wsRef.current?.readyState === WebSocket.OPEN),
    });
  }, [isMockMode]);

  /* Connect on mount (real mode only) */
  useEffect(() => {
    if (!isMockMode) {
      connect();
    } else {
      setState((prev) => ({ ...prev, isConnected: true }));
    }

    return () => {
      wsRef.current?.close();
      clearTimeout(reconnectTimer.current);
      mockTimers.current.forEach(clearTimeout);
    };
  }, [connect, isMockMode]);

  return {
    state,
    startInvestigation,
    resetInvestigation,
  };
}
