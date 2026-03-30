import { useEffect, useRef, useState } from 'react';
import { motion } from 'framer-motion';
import type { PipelineStage, Hypothesis } from '@/hooks/useInvestigation';
import PipelineNode from './PipelineNode';
import { Terminal } from 'lucide-react';

interface TerminalSidebarProps {
  logs: string[];
  currentStage: PipelineStage;
  hypotheses: Hypothesis[];
}

const PIPELINE_NODES = [
  { label: 'DECOMPOSE', stage: 'scanning' },
  { label: 'HYPOTHESIZE', stage: 'generating' },
  { label: 'RETRIEVE', stage: 'retrieving' },
  { label: 'SCORE', stage: 'scoring' },
  { label: 'CONCLUDE', stage: 'concluded' },
];

const TypewriterText = ({ text }: { text: string }) => {
  return (
    <motion.span
      initial="hidden"
      animate="visible"
      transition={{ staggerChildren: 0.01 }}
    >
      {text.split('').map((char, index) => (
        <motion.span
          key={index}
          variants={{
            hidden: { opacity: 0, display: 'none' },
            visible: { opacity: 1, display: 'inline' },
          }}
        >
          {char}
        </motion.span>
      ))}
    </motion.span>
  );
};

export default function TerminalSidebar({
  logs,
  currentStage,
  hypotheses,
}: TerminalSidebarProps) {
  const logsEndRef = useRef<HTMLDivElement>(null);
  
  // Create state to hold a deterministic timestamp per log line so they don't jump on re-renders
  const [logTimestamps, setLogTimestamps] = useState<Record<number, string>>({});

  useEffect(() => {
    logs.forEach((_, i) => {
      setLogTimestamps(prev => {
        if (prev[i]) return prev;
        const date = new Date();
        const str = `[${date.getHours().toString().padStart(2, '0')}:${date.getMinutes().toString().padStart(2, '0')}:${date.getSeconds().toString().padStart(2, '0')}]`;
        return { ...prev, [i]: str };
      });
    });
    logsEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [logs]);

  // Compute stats
  const evCount = hypotheses.reduce((acc, h) => acc + h.evidence.length, 0);
  const sources = new Set(hypotheses.flatMap(h => h.evidence.map(e => e.domain_tag))).size;

  return (
    <motion.aside
      className="w-full h-full flex flex-col p-4 bg-dark-matter"
      initial={{ x: -300, opacity: 0 }}
      animate={{ x: 0, opacity: 1 }}
      transition={{ duration: 0.5, ease: 'easeOut' }}
    >
      {/* ── Header ── */}
      <div className="flex items-center gap-2 pb-4 border-b border-border-glass mb-4">
        <Terminal className="w-5 h-5 text-electric-cyan" />
        <span className="font-mono text-[14px] font-bold tracking-[0.2em] text-electric-cyan">
          OPS TRACKER
        </span>
      </div>

      {/* ── Pipeline Nodes ── */}
      <div className="flex flex-col flex-shrink-0">
        {PIPELINE_NODES.map((node, i) => (
          <PipelineNode
            key={node.stage}
            label={node.label}
            stage={node.stage}
            currentStage={currentStage}
            index={i}
          />
        ))}
      </div>

      {/* ── Terminal Feed ── */}
      <div className="flex-1 mt-6 border border-electric-cyan/10 bg-black/40 p-3 overflow-y-auto min-h-[200px] max-h-[300px]">
        <div className="flex items-center gap-2 mb-3 sticky top-0 bg-black/80 p-1">
          <div className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse-glow" />
          <span className="font-mono text-[11px] font-bold tracking-[0.2em] text-electric-cyan">
            TERMINAL // STDOUT
          </span>
        </div>

        <div className="font-mono text-[12px] leading-relaxed space-y-1">
          {logs.length === 0 && (
            <p className="text-ghost-faint animate-typewriter border-r-2 border-electric-cyan pr-1 inline-block">
              AWAITING STREAM...
            </p>
          )}

          {logs.map((log, i) => {
            const isCritical = log.includes('✓') || log.includes('COMPLETE') || log.includes('NEW');
            return (
              <motion.div
                key={`${i}-${log.slice(0, 10)}`}
                className="flex items-start gap-2"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ duration: 0.1 }}
              >
                <span className="text-electric-cyan/40 shrink-0">
                  {logTimestamps[i] || '[--:--:--]'}
                </span>
                <p
                   className={
                     log.includes('✗')
                       ? 'text-crimson-burn font-bold'
                       : isCritical
                         ? 'text-electric-cyan font-bold brightness-125'
                         : log.includes('═')
                           ? 'text-toxic-violet/60'
                           : 'text-electric-cyan/70'
                   }
                >
                  <motion.span
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ staggerChildren: 0.03, delayChildren: 0.1 }}
                  >
                    <TypewriterText text={log} />
                  </motion.span>
                </p>
              </motion.div>
            );
          })}
          <div ref={logsEndRef} />
        </div>
      </div>

      {/* ── Footer Stats ── */}
      <div className="mt-4 pt-4 border-t border-border-glass grid grid-cols-2 gap-y-3 gap-x-2">
         <div className="flex flex-col">
            <span className="font-mono text-[10px] tracking-widest text-[#B0B0C8]">HYPOTHESES</span>
            <span className="font-mono text-[18px] text-electric-cyan font-bold">{hypotheses.length}</span>
         </div>
         <div className="flex flex-col">
            <span className="font-mono text-[10px] tracking-widest text-[#B0B0C8]">EVIDENCE</span>
            <span className="font-mono text-[18px] text-electric-cyan font-bold">{evCount}</span>
         </div>
         <div className="flex flex-col col-span-2">
            <span className="font-mono text-[10px] tracking-widest text-[#B0B0C8]">SOURCE DOMAINS</span>
            <span className="font-mono text-[18px] text-electric-cyan font-bold">{sources}</span>
         </div>
      </div>
    </motion.aside>
  );
}
