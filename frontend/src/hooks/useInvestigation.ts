import { useCallback, useEffect, useRef, useState } from 'react';

/* ============================================================
   TYPE DEFINITIONS
   ============================================================ */

export type PipelineStage = 
  | 'idle' | 'scanning' | 'generating' 
  | 'retrieving' | 'scoring' | 'concluded';

export type HypothesisStatus = 
  | 'pending' | 'active' | 'survivor' | 'eliminated';

export interface Evidence {
  id: string;
  url: string;
  domain: string;
  title: string;
  excerpt: string;
  favicon?: string;
}

export interface Hypothesis {
  id: string;
  title: string;
  body: string;
  confidence: number;
  status: HypothesisStatus;
  evidence: Evidence[];
  eliminationReason?: string;
  retrievedAt?: string;
}

export interface LogEntry {
  timestamp: string;
  message: string;
  level: 'info' | 'warn' | 'error' | 'success';
}

export interface InvestigationState {
  stage: PipelineStage;
  query: string;
  hypotheses: Hypothesis[];
  logs: LogEntry[];
  verdict: {
    status: 'CASE RESOLVED' | 'INCONCLUSIVE' | null;
    narrative: string;
    caveats: string[];
    sources: string[];
    confidence: number;
  } | null;
  counters: { hypotheses: number; evidence: number; sources: number; };
  startedAt: Date | null;
  isConnected: boolean;
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
    body: 'Evidence suggests systematic suppression of information by state intelligence agencies, including classified document redactions and witness intimidation campaigns.',
    confidence: 0,
    status: 'active',
    evidence: [],
  },
  {
    id: 'hyp_002',
    title: 'Natural Phenomenon',
    body: 'Environmental and atmospheric conditions created a rare convergence of natural events that explain all observed anomalies without requiring human intervention.',
    confidence: 0,
    status: 'active',
    evidence: [],
  },
  {
    id: 'hyp_003',
    title: 'Mass Hysteria Event',
    body: 'Social contagion and media amplification transformed ordinary events into an extraordinary narrative through collective misperception and confirmation bias.',
    confidence: 0,
    status: 'active',
    evidence: [],
  },
  {
    id: 'hyp_004',
    title: 'Unknown Technology',
    body: 'Physical evidence and expert testimony point to technology beyond publicly known capabilities, possibly from classified military programs or unknown origin.',
    confidence: 0,
    status: 'active',
    evidence: [],
  },
];

const MOCK_EVIDENCE: Evidence[] = [
  {
    id: 'ev_001',
    url: 'https://en.wikipedia.org/wiki/example',
    title: 'Wikipedia',
    excerpt: 'Declassified documents from 1967 reveal that intelligence agencies actively monitored and suppressed civilian reports...',
    domain: 'wikipedia.org',
  },
  {
    id: 'ev_002',
    url: 'https://www.nature.com/articles/example',
    title: 'Nature',
    excerpt: 'Atmospheric inversion layers combined with temperature gradients can produce optical phenomena consistent with reported observations...',
    domain: 'nature.com',
  },
  {
    id: 'ev_003',
    url: 'https://www.history.com/example',
    title: 'History.com',
    excerpt: 'Similar patterns of mass sighting events have been documented throughout history, typically following periods of social anxiety...',
    domain: 'history.com',
  },
  {
    id: 'ev_004',
    url: 'https://arxiv.org/abs/example',
    title: 'ArXiv',
    excerpt: 'Analysis of radar data shows objects exhibiting flight characteristics inconsistent with known aircraft or atmospheric phenomena...',
    domain: 'arxiv.org',
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

function getMockConclusion() {
  return {
    status: 'CASE RESOLVED' as const,
    narrative: 'After systematic analysis of all available evidence, the investigation concludes that the Government Cover-Up hypothesis (H-001) presents the strongest evidence-backed explanation. Declassified documents and verified witness testimony provide substantial corroboration, while competing hypotheses lacked sufficient evidentiary support to survive rigorous scoring. Two hypotheses were eliminated for confidence scores below the 0.35 threshold.',
    caveats: [
      'Limited access to classified primary sources',
      'Historical accounts may contain factual inaccuracies',
      'Correlation does not imply causation in pattern analysis',
    ],
    sources: [
      'en.wikipedia.org',
      'nature.com',
      'history.com',
      'arxiv.org',
    ],
    confidence: 0.72,
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
  query: '',
  hypotheses: [],
  logs: [],
  verdict: null,
  counters: { hypotheses: 0, evidence: 0, sources: 0 },
  startedAt: null,
  isConnected: false,
};

function createLogEntry(message: string, isError = false, isWarn = false, isSuccess = false): LogEntry {
  const now = new Date();
  const timeStr = `[${now.getHours().toString().padStart(2, '0')}:${now.getMinutes().toString().padStart(2, '0')}:${now.getSeconds().toString().padStart(2, '0')}]`;
  let level: LogEntry['level'] = 'info';
  if (isError) level = 'error';
  else if (isWarn) level = 'warn';
  else if (isSuccess) level = 'success';

  return { timestamp: timeStr, message, level };
}

export function useInvestigation() {
  const [state, setState] = useState<InvestigationState>(INITIAL_STATE);
  const wsRef = useRef<WebSocket | null>(null);
  const logsBuffer = useRef(new CircularBuffer<LogEntry>(200));
  const reconnectAttempts = useRef(0);
  const reconnectTimer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);
  const mockTimers = useRef<ReturnType<typeof setTimeout>[]>([]);

  const isMockMode = import.meta.env.VITE_MOCK === 'true';

  /* Push log lines and update state */
  const addLogs = useCallback((messages: string[], level: LogEntry['level'] = 'info') => {
    const entries = messages.map(msg => createLogEntry(
      msg, 
      level === 'error' || msg.includes('ERROR') || msg.includes('✗'),
      level === 'warn',
      level === 'success' || msg.includes('✓') || msg.includes('COMPLETE') || msg.includes('SURVIVING') || msg.includes('HIT')
    ));
    logsBuffer.current.pushMany(entries);
    setState((prev) => ({
      ...prev,
      logs: logsBuffer.current.getAll(),
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
            setState((prev) => ({ ...prev, stage: 'scanning', query: data.mystery as string, startedAt: new Date() }));
            break;

          case 'node_complete': {
            const nodeName = data.node as string;
            const nodeData = data.data as Record<string, unknown>;

            addLogs([`> NODE COMPLETE: ${nodeName.toUpperCase()}`], 'success');

            setState((prev) => {
              const newStage = NODE_TO_STAGE[nodeName] ?? prev.stage;
              const updated = { ...prev, stage: newStage };

              if (nodeName === 'hypothesize' && nodeData.hypotheses) {
                updated.hypotheses = (
                  nodeData.hypotheses as Array<Record<string, unknown>>
                ).map((h) => ({
                  id: h.id as string,
                  title: h.title as string,
                  body: h.description as string,
                  confidence: h.plausibility_score as number,
                  status: (h.status === 'surviving' ? 'survivor' : h.status ?? 'active') as HypothesisStatus,
                  eliminationReason: h.elimination_reason as string | undefined,
                  evidence: [],
                }));
                updated.counters = { ...updated.counters, hypotheses: updated.hypotheses.length };
              }

              if (nodeName === 'retrieve_evidence' && nodeData.evidence) {
                const evidenceMap = nodeData.evidence as Record<string, Array<Record<string, unknown>>>;
                let evidenceCount = 0;
                const sourceDomains = new Set<string>();

                updated.hypotheses = prev.hypotheses.map((h) => {
                  const rawEvs = evidenceMap[h.id] || [];
                  const newEvs = rawEvs.map(e => {
                    evidenceCount++;
                    if (e.source_domain) sourceDomains.add(e.source_domain as string);
                    return {
                      id: e.id as string,
                      url: e.source_url as string,
                      title: e.source_name as string,
                      excerpt: e.text as string,
                      domain: (e.source_domain || e.domain_tag) as string,
                      favicon: e.favicon as string | undefined
                    };
                  });
                  return {
                    ...h,
                    evidence: newEvs,
                  };
                });
                updated.counters = { ...updated.counters, evidence: evidenceCount, sources: sourceDomains.size };
              }

              if (nodeName === 'score_and_eliminate' && nodeData.hypotheses) {
                const scoredHypotheses = nodeData.hypotheses as Array<Record<string, unknown>>;
                const scoredMap = new Map(
                  ((nodeData.scored_hypotheses as Array<Record<string, unknown>>) ?? []).map((sh) => [sh.hypothesis_id as string, sh.confidence_score as number])
                );

                updated.hypotheses = prev.hypotheses.map((h) => {
                  const scored = scoredHypotheses.find((sh) => sh.id === h.id);
                  let newStatus = (scored?.status as string) ?? h.status;
                  if (newStatus === 'surviving') newStatus = 'survivor';

                  return {
                    ...h,
                    status: newStatus as HypothesisStatus,
                    eliminationReason: (scored?.elimination_reason as string) ?? h.eliminationReason,
                    confidence: scoredMap.get(h.id) ?? h.confidence,
                  };
                });
              }

              if (nodeName === 'conclude' && nodeData.conclusion) {
                const rawCon = nodeData.conclusion as Record<string, unknown>;
                updated.verdict = {
                  status: rawCon.confidence_label === 'High' ? 'CASE RESOLVED' : 'INCONCLUSIVE',
                  narrative: rawCon.summary as string,
                  caveats: rawCon.caveats as string[],
                  sources: rawCon.all_sources as string[],
                  confidence: rawCon.overall_confidence as number,
                };
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
            ], 'warn');
            break;

          case 'investigation_complete':
            addLogs(['> ═══════════════════════════════', '> INVESTIGATION COMPLETE', '> ═══════════════════════════════'], 'success');
            break;

          case 'error':
            addLogs([`> ✗ ERROR: ${data.message as string}`], 'error');
            setState((prev) => ({
              ...prev,
              error: data.message as string,
            }));
            break;
        }
      } catch {
        addLogs(['> ✗ FAILED TO PARSE SERVER MESSAGE'], 'error');
      }
    },
    [addLogs]
  );

  /* Connect to WebSocket */
  const connect = useCallback(() => {
    if (isMockMode) return;

    const rawUrl = import.meta.env.VITE_WS_URL || 'localhost:8000';
    const isFullUrl = rawUrl.startsWith('ws://') || rawUrl.startsWith('wss://');
    const secureProto = window.location.protocol === 'https:' ? 'wss://' : 'ws://';
    const finalUrl = isFullUrl ? `${rawUrl}/ws/investigate` : `${secureProto}${rawUrl}/ws/investigate`;
    
    const ws = new WebSocket(finalUrl);

    ws.onopen = () => {
      setState((prev) => ({ ...prev, isConnected: true }));
      reconnectAttempts.current = 0;
      addLogs(['> WEBSOCKET CONNECTED']);
    };

    ws.onmessage = handleMessage;

    ws.onclose = () => {
      setState((prev) => ({ ...prev, isConnected: false }));
      addLogs(['> WEBSOCKET DISCONNECTED'], 'warn');

      const delay = Math.min(1000 * Math.pow(2, reconnectAttempts.current), 30000);
      reconnectAttempts.current++;
      reconnectTimer.current = setTimeout(connect, delay);
    };

    ws.onerror = () => {
      addLogs(['> ✗ WEBSOCKET ERROR'], 'error');
    };

    wsRef.current = ws;
  }, [isMockMode, handleMessage, addLogs]);

  /* Start investigation */
  const startInvestigation = useCallback(
    (query: string) => {
      logsBuffer.current.clear();
      setState(prev => ({
        ...INITIAL_STATE,
        isConnected: isMockMode || prev.isConnected,
        query,
        startedAt: new Date(),
        logs: [],
      }));

      if (isMockMode) {
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
                    confidence: 0.5,
                  }));
                  updated.counters = { ...updated.counters, hypotheses: MOCK_HYPOTHESES.length };
                }

                if (node === 'retrieve_evidence') {
                  const evMap: Record<string, Evidence[]> = {
                    hyp_001: [MOCK_EVIDENCE[0] as Evidence, MOCK_EVIDENCE[1] as Evidence],
                    hyp_002: [MOCK_EVIDENCE[1] as Evidence],
                    hyp_003: [MOCK_EVIDENCE[2] as Evidence],
                    hyp_004: [MOCK_EVIDENCE[3] as Evidence],
                  };
                  let evTotal = 0;
                  updated.hypotheses = prev.hypotheses.map((h) => {
                    evTotal += (evMap[h.id]?.length || 0);
                    return {
                      ...h,
                      evidence: evMap[h.id] || [],
                    };
                  });
                  updated.counters = { ...updated.counters, evidence: evTotal, sources: 4 };
                }

                if (node === 'score_and_eliminate') {
                  updated.hypotheses = prev.hypotheses.map((h) => {
                    if (h.id === 'hyp_003') {
                      return { ...h, status: 'eliminated' as const, confidence: 0.28, eliminationReason: 'Insufficient evidence to support mass hysteria as primary explanation' };
                    }
                    if (h.id === 'hyp_004') {
                      return { ...h, status: 'eliminated' as const, confidence: 0.31, eliminationReason: 'No verifiable physical evidence of unknown technology' };
                    }
                    return { ...h, status: 'survivor' as const, confidence: h.id === 'hyp_001' ? 0.72 : 0.58 };
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
        if (wsRef.current?.readyState === WebSocket.OPEN) {
          wsRef.current.send(JSON.stringify({ mystery: query }));
        } else {
          addLogs(['> ✗ WEBSOCKET NOT CONNECTED'], 'error');
        }
      }
    },
    [isMockMode, addLogs]
  );

  const resetInvestigation = useCallback(() => {
    logsBuffer.current.clear();
    mockTimers.current.forEach(clearTimeout);
    mockTimers.current = [];
    setState({
      ...INITIAL_STATE,
      isConnected: isMockMode || (wsRef.current?.readyState === WebSocket.OPEN),
    });
  }, [isMockMode]);

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
