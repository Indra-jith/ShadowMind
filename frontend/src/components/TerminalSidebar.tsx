import React, { useEffect, useRef, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { CheckCheck, X } from 'lucide-react';
import { InvestigationState, PipelineStage } from '../hooks/useInvestigation';

const PIPELINE_NODES = [
  { id: 'scanning', number: '01', name: 'NEURAL DECOMPOSITION', desc: 'Parsing query into investigation vectors...' },
  { id: 'generating', number: '02', name: 'HYPOTHESIS GENERATION', desc: 'Applying adversarial diversity filters...' },
  { id: 'retrieving', number: '03', name: 'EVIDENCE RETRIEVAL', desc: 'Querying vector stores and live web sources...' },
  { id: 'scoring', number: '04', name: 'LLM SCORING', desc: 'Evaluating hypotheses against retrieved evidence...' },
  { id: 'concluded', number: '05', name: 'VERDICT SYNTHESIS', desc: 'Aggregating surviving evidence for final conclusion...' }
];

const STAGE_ORDER = ['idle', 'scanning', 'generating', 'retrieving', 'scoring', 'concluded'];

function getStatus(stageId: string, currentStage: PipelineStage) {
  const currentIndex = STAGE_ORDER.indexOf(currentStage);
  const nodeIndex = STAGE_ORDER.indexOf(stageId);
  
  if (currentIndex === 0) return 'PENDING';
  if (nodeIndex < currentIndex) return 'COMPLETE';
  if (nodeIndex === currentIndex) return 'ACTIVE';
  return 'PENDING';
}

const TypewriterText: React.FC<{ text: string }> = ({ text }) => {
  const [displayed, setDisplayed] = useState('');
  
  useEffect(() => {
    let i = 0;
    setDisplayed('');
    const timer = setInterval(() => {
      setDisplayed(text.slice(0, i + 1));
      i++;
      if (i >= text.length) clearInterval(timer);
    }, 30);
    return () => clearInterval(timer);
  }, [text]);

  return <span>{displayed}</span>;
}

const CounterCell = ({ label, value }: { label: string, value: number }) => (
  <div className="flex flex-col items-center justify-center p-[10px] w-1/3">
    <span 
      className="text-center text-[10px] tracking-[0.2em] text-[rgba(112,112,160,0.6)]"
      style={{ fontFamily: 'var(--font-mono)' }}
    >
      {label}
    </span>
    <motion.span
      key={value}
      initial={{ scale: 1 }}
      animate={{ scale: [1, 1.3, 1] }}
      transition={{ type: 'spring', stiffness: 300 }}
      className="mt-1 text-[28px] text-[var(--color-cyan)] leading-none"
      style={{ fontFamily: 'var(--font-display)' }}
    >
      {value}
    </motion.span>
  </div>
);

export default function TerminalSidebar({ state }: { state: InvestigationState }) {
  const logEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [state.logs]);

  return (
    <div 
      className="flex h-full w-[240px] md:w-[320px] flex-col border-r border-[rgba(0,245,255,0.06)] bg-[rgba(6,0,16,0.4)]"
      style={{ backdropFilter: 'blur(8px)' }}
    >
      {/* SECTION A: Pipeline Tracker */}
      <div className="flex-shrink-0">
        <div className="border-b border-[rgba(0,245,255,0.06)] px-[20px] py-[16px]">
          <span 
            className="text-[11px] tracking-[0.2em] text-[rgba(0,245,255,0.5)]"
            style={{ fontFamily: 'var(--font-mono)' }}
          >
            PIPELINE STATUS
          </span>
        </div>

        <div className="flex flex-col">
          {PIPELINE_NODES.map((node) => {
            const status = getStatus(node.id, state.stage);
            const isActive = status === 'ACTIVE';
            const isComplete = status === 'COMPLETE';
            const isPending = status === 'PENDING';

            let bgColor = 'transparent';
            let borderColor = 'rgba(112,112,160,0.3)';
            let statusColor = 'rgba(112,112,160,0.6)';

            if (isActive) {
              bgColor = 'rgba(0,245,255,0.1)';
              borderColor = 'var(--color-cyan)';
              statusColor = 'var(--color-cyan)';
            } else if (isComplete) {
              bgColor = 'rgba(0,255,136,0.15)';
              borderColor = 'var(--color-green)';
              statusColor = 'var(--color-green)';
            }

            return (
              <div 
                key={node.id}
                className="flex items-start gap-[14px] border-b border-[rgba(255,255,255,0.03)] px-[20px] py-[14px] min-h-[72px] transition-all duration-400"
              >
                {/* Status Indicator */}
                <div 
                  className="mt-1 flex h-[18px] w-[18px] flex-shrink-0 items-center justify-center rounded-full border-[1.5px]"
                  style={{
                    backgroundColor: bgColor,
                    borderColor: borderColor,
                    boxShadow: isActive ? '0 0 12px rgba(0,245,255,0.5)' : 'none',
                    animation: isActive ? 'pulse-ring 1.5s infinite' : 'none'
                  }}
                >
                  {isComplete && <CheckCheck className="h-[10px] w-[10px] text-[#00FF88]" />}
                </div>

                {/* Node Content */}
                <div className="flex flex-col flex-1 w-full overflow-hidden">
                  <div className="flex items-baseline justify-between w-full">
                    <span 
                      className="text-[10px] tracking-[0.15em] flex-shrink-0"
                      style={{ fontFamily: 'var(--font-mono)', color: statusColor }}
                    >
                      {node.number} <span className="text-[14px] font-semibold text-[#E8E8F0] ml-1 uppercase" style={{ fontFamily: 'var(--font-body)' }}>{node.name}</span>
                    </span>
                  </div>

                  {/* Active detailed states */}
                  <AnimatePresence>
                    {isActive && (
                      <motion.div
                        initial={{ height: 0, opacity: 0 }}
                        animate={{ height: 'auto', opacity: 1 }}
                        exit={{ height: 0, opacity: 0 }}
                        className="mt-2 overflow-hidden flex flex-col"
                      >
                        <span 
                          className="text-[11px] leading-[1.5] text-[rgba(112,112,160,0.7)]"
                          style={{ fontFamily: 'var(--font-mono)' }}
                        >
                          {node.desc}
                        </span>
                        
                        {/* Progress Bar */}
                        <div className="mt-3 h-[2px] w-full rounded-[1px] bg-[rgba(255,255,255,0.05)] overflow-hidden">
                          <motion.div
                            initial={{ width: '0%' }}
                            animate={{ width: '100%' }}
                            transition={{ duration: 25, ease: "linear" }} /* faux progress */
                            className="h-full bg-gradient-to-r from-[rgba(0,245,255,0.2)] to-[var(--color-cyan)]"
                          />
                        </div>
                      </motion.div>
                    )}
                  </AnimatePresence>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* SECTION B: Counter Strip */}
      <div 
        className="flex flex-shrink-0 border-y border-[rgba(255,255,255,0.04)]"
        style={{ backgroundColor: 'rgba(0,0,0,0.2)' }}
      >
        <CounterCell label="HYPOTHESES" value={state.counters?.hypotheses || 0} />
        <div className="w-[1px] bg-[rgba(255,255,255,0.04)]" />
        <CounterCell label="EVIDENCE" value={state.counters?.evidence || 0} />
        <div className="w-[1px] bg-[rgba(255,255,255,0.04)]" />
        <CounterCell label="SOURCES" value={state.counters?.sources || 0} />
      </div>

      {/* SECTION C: Terminal STDOUT Feed */}
      <div className="flex flex-1 flex-col overflow-hidden">
        <div className="flex items-center justify-between border-b border-[rgba(0,245,255,0.06)] px-[20px] py-[16px]">
          <span 
            className="text-[11px] tracking-[0.2em] text-[rgba(0,245,255,0.5)] uppercase"
            style={{ fontFamily: 'var(--font-mono)' }}
          >
            SYSTEM OUTPUT
          </span>
          {state.stage !== 'idle' && state.stage !== 'concluded' && (
            <div className="flex items-center gap-1.5">
              <div className="h-2 w-2 rounded-full bg-[var(--color-green)] opacity-80" style={{ animation: 'pulse-ring 1s infinite' }} />
              <span className="text-[10px] text-[var(--color-green)]" style={{ fontFamily: 'var(--font-mono)' }}>LIVE</span>
            </div>
          )}
        </div>
        
        <div 
          className="flex-1 overflow-y-scroll px-[16px] py-[12px] scrollbar-thin scrollbar-track-transparent scrollbar-thumb-[rgba(0,245,255,0.2)]"
          style={{ backgroundColor: 'rgba(0,0,0,0.2)' }}
        >
          {state.logs.map((log, i) => {
            const isHighPri = log.level === 'success';
            const isError = log.level === 'error';
            const isWarn = log.level === 'warn';

            let msgColor = 'rgba(232,232,240,0.6)';
            let fw = 400;
            if (isHighPri) { msgColor = 'var(--color-cyan)'; fw = 700; }
            if (isError) msgColor = 'var(--color-crimson)';
            if (isWarn) msgColor = 'var(--color-amber)';

            return (
              <div key={i} className="mb-[4px] flex gap-[10px]">
                <span 
                  className="flex-shrink-0 text-[11px] text-[rgba(0,245,255,0.3)] mt-0.5"
                  style={{ fontFamily: 'var(--font-mono)' }}
                >
                  {log.timestamp}
                </span>
                <span 
                  className="text-[12px] leading-[1.5]"
                  style={{ fontFamily: 'var(--font-mono)', color: msgColor, fontWeight: fw }}
                >
                  {isHighPri && <span className="mr-1 inline-block">▸</span>}
                  <TypewriterText text={log.message} />
                </span>
              </div>
            );
          })}
          <div ref={logEndRef} className="h-2" />
        </div>
      </div>
    </div>
  );
}
