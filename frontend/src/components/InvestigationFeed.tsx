import React, { useEffect, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { InvestigationState } from '../hooks/useInvestigation';
import HypothesisCard from './HypothesisCard';
import VerdictBlock from './VerdictBlock';

interface InvestigationFeedProps {
  state: InvestigationState;
  onReset: () => void;
}

const Timer: React.FC<{ startedAt: Date | null, stage: string }> = ({ startedAt, stage }) => {
  const [elapsed, setElapsed] = useState(0);

  useEffect(() => {
    if (!startedAt) {
      setElapsed(0);
      return;
    }
    
    if (stage === 'concluded') return;

    const interval = setInterval(() => {
      setElapsed(Math.floor((new Date().getTime() - startedAt.getTime()) / 1000));
    }, 1000);

    return () => clearInterval(interval);
  }, [startedAt, stage]);

  if (!startedAt) return <span>00:00:00</span>;

  const h = Math.floor(elapsed / 3600).toString().padStart(2, '0');
  const m = Math.floor((elapsed % 3600) / 60).toString().padStart(2, '0');
  const s = (elapsed % 60).toString().padStart(2, '0');

  return <span>T+{h}:{m}:{s}</span>;
}

export default function InvestigationFeed({ state, onReset }: InvestigationFeedProps) {
  // Sort hypotheses: survivors first, then by confidence desc
  const sortedHypotheses = [...state.hypotheses].sort((a, b) => {
    if (a.status === 'survivor' && b.status !== 'survivor') return -1;
    if (b.status === 'survivor' && a.status !== 'survivor') return 1;
    if (a.status === 'eliminated' && b.status !== 'eliminated') return 1;
    if (b.status === 'eliminated' && a.status !== 'eliminated') return -1;
    return (b.confidence || 0) - (a.confidence || 0);
  });

  return (
    <div className="flex-1 w-full overflow-y-auto px-[16px] md:px-[40px] pt-[32px] pb-[80px] scrollbar-thin scrollbar-track-transparent scrollbar-thumb-[rgba(0,245,255,0.4)] hover:scrollbar-thumb-[rgba(0,245,255,1)]">
      
      {/* QUERY BANNER */}
      <div 
        className="mb-[32px] rounded-[6px] border border-[rgba(0,245,255,0.08)] bg-[rgba(0,0,0,0.2)] p-[20px_24px] flex flex-col md:flex-row justify-between md:items-start items-start md:items-center gap-4"
      >
        <div>
          <span 
            className="text-[11px] tracking-[0.2em] text-[rgba(0,245,255,0.5)]"
            style={{ fontFamily: 'var(--font-mono)' }}
          >
            ◌ ACTIVE INVESTIGATION
          </span>
          <h2 
            className="mt-[8px] text-[20px] font-semibold text-[#E8E8F0]"
            style={{ fontFamily: 'var(--font-body)' }}
          >
            {state.query || 'Awaiting target designation...'}
          </h2>
        </div>
        
        <div 
          className="flex flex-col items-start md:items-end text-[12px] text-[rgba(112,112,160,0.5)] leading-tight"
          style={{ fontFamily: 'var(--font-mono)' }}
        >
          {state.startedAt && (
            <span>STARTED: {state.startedAt.toLocaleTimeString()}</span>
          )}
          <Timer startedAt={state.startedAt} stage={state.stage} />
        </div>
      </div>

      {/* HYPOTHESIS CARDS REGION */}
      {sortedHypotheses.length > 0 && (
        <div className="mb-[16px] flex items-center gap-[12px]">
          <div className="flex-1 h-[1px] bg-[rgba(0,245,255,0.12)]" />
          <span 
            className="text-[11px] tracking-[0.25em] text-[rgba(0,245,255,0.4)] uppercase"
            style={{ fontFamily: 'var(--font-mono)' }}
          >
            ─── HYPOTHESIS MATRIX ──
          </span>
          <div className="flex-1 h-[1px] bg-[rgba(0,245,255,0.12)]" />
        </div>
      )}

      <motion.div layout className="flex flex-col gap-[20px] relative">
        <AnimatePresence>
          {sortedHypotheses.map((h, i) => (
            <motion.div
              layout
              key={h.id}
              layoutId={h.id}
              initial={{ opacity: 0, y: -20, filter: 'blur(4px)' }}
              animate={{ opacity: 1, y: 0, filter: 'blur(0)' }}
              transition={{ delay: i * 0.12, type: 'spring', stiffness: 200, damping: 25 }}
              className="w-full"
            >
              <HypothesisCard hypothesis={h} index={i} />
            </motion.div>
          ))}
        </AnimatePresence>
      </motion.div>

      {/* VERDICT SECTION */}
      <AnimatePresence>
        {state.stage === 'concluded' && state.verdict && (
          <motion.div 
            initial={{ opacity: 0, y: 40 }} 
            animate={{ opacity: 1, y: 0 }} 
            transition={{ type: 'spring', stiffness: 150, damping: 25, delay: 0.5 }}
          >
            <VerdictBlock verdict={state.verdict} onReset={onReset} />
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
