import React, { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import { X } from 'lucide-react';
import { useInvestigation } from '../hooks/useInvestigation';
import TerminalSidebar from './TerminalSidebar';
import InvestigationFeed from './InvestigationFeed';

interface DashboardProps {
  initialQuery?: string;
  onBack: () => void;
}

const StatusBar: React.FC = () => {
  return (
    <div 
      className="fixed bottom-0 z-50 flex w-full items-center justify-between px-4 md:px-[120px] h-[48px]"
      style={{
        background: 'rgba(6,0,16,0.85)',
        backdropFilter: 'blur(16px)',
        borderTop: '1px solid rgba(0,245,255,0.07)',
      }}
    >
      <div className="flex items-center gap-4 md:gap-8">
        <div className="flex items-center gap-2">
          <div className="h-2 w-2 rounded-full bg-[#00FF88]" style={{ animation: 'pulse-ring 2s infinite' }} />
          <span className="text-[12px] uppercase text-[rgba(232,232,240,0.55)]" style={{ fontFamily: 'var(--font-mono)', letterSpacing: '0.1em' }}>SYS ONLINE</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="h-2 w-2 rounded-full bg-[#00F5FF]" style={{ animation: 'pulse-ring 2s infinite 0.5s' }} />
          <span className="text-[12px] uppercase text-[rgba(232,232,240,0.55)]" style={{ fontFamily: 'var(--font-mono)', letterSpacing: '0.1em' }}>NEURAL NET READY</span>
        </div>
        <div className="hidden md:flex items-center gap-2">
          <span className="text-[#FFB300]">◈</span>
          <span className="text-[12px] uppercase text-[rgba(232,232,240,0.55)]" style={{ fontFamily: 'var(--font-mono)', letterSpacing: '0.1em' }}>THREAT LEVEL: UNKNOWN</span>
        </div>
      </div>
      <div className="flex items-center gap-3">
        <span className="text-[12px] text-[rgba(0,245,255,0.5)]" style={{ fontFamily: 'var(--font-mono)' }}>NODES ACTIVE: 5 / 5</span>
        <div className="flex h-[24px] items-end gap-[4px] overflow-hidden">
          {[1, 2, 3, 4, 5].map((i) => (
            <motion.div
              key={i}
              initial={{ y: 48 }}
              animate={{ y: 0 }}
              transition={{ delay: 2.1 + (i * 0.1), type: 'spring', stiffness: 300, damping: 30 }}
              className="w-[2px] bg-[rgba(0,245,255,0.4)]"
              style={{ 
                height: `${30 + Math.random() * 60}%`,
                animation: `float-bar ${1.5 + Math.random()}s infinite ease-in-out alternate ${i * 0.2}s`
              }}
            />
          ))}
        </div>
      </div>
    </div>
  );
};

export default function Dashboard({ initialQuery = '', onBack }: DashboardProps) {
  const { state, startInvestigation, resetInvestigation } = useInvestigation();
  const [showCursor, setShowCursor] = useState(true);

  useEffect(() => {
    // Wait for the websocket to connect before attempting to start the investigation
    if (state.stage === 'idle' && !state.query && state.isConnected) {
      if (initialQuery) {
        startInvestigation(initialQuery);
      }
    }
  }, [state.stage, state.query, state.isConnected, startInvestigation, initialQuery]);

  useEffect(() => {
    const i = setInterval(() => setShowCursor(c => !c), 500);
    return () => clearInterval(i);
  }, []);

  const handleReset = () => {
    resetInvestigation();
    onBack();
  };

  const getStageColor = (stage: string) => {
    switch (stage) {
      case 'scanning': return 'var(--color-cyan)';
      case 'generating': return 'var(--color-violet)';
      case 'retrieving': return 'var(--color-cyan)';
      case 'scoring': return 'var(--color-amber)';
      case 'concluded': return 'var(--color-green)';
      default: return 'var(--color-text-muted)';
    }
  };

  const stageColor = getStageColor(state.stage);

  return (
    <div className="min-h-screen w-full overflow-hidden bg-[var(--color-void)] text-[#E8E8F0]">
      {/* DASHBOARD HEADER */}
      <header 
        className="fixed top-0 z-50 flex h-[64px] w-full items-center justify-between px-4 md:px-8"
        style={{
          background: 'rgba(6,0,16,0.92)',
          backdropFilter: 'blur(20px)',
          borderBottom: '1px solid rgba(0,245,255,0.08)'
        }}
      >
        <div className="flex items-center">
          <div className="flex items-center text-[22px] tracking-[0.2em]" style={{ fontFamily: 'var(--font-display)' }}>
            <span className="mr-2 text-[var(--color-cyan)] drop-shadow-[0_0_6px_var(--color-cyan)]">◈</span>
            SHADOWMIND
          </div>
          <div className="mx-[20px] h-[20px] w-[1px] bg-[rgba(0,245,255,0.15)]" />
        </div>

        <div className="hidden flex-1 items-center md:flex max-w-[480px]">
          <span 
            className="mr-2 text-[11px] text-[rgba(0,245,255,0.5)] tracking-[0.15em]"
            style={{ fontFamily: 'var(--font-mono)' }}
          >
            ◌ INVESTIGATING:
          </span>
          <span 
            className="truncate text-[14px] font-medium"
            style={{ fontFamily: 'var(--font-mono)' }}
          >
            {state.query || 'AWAITING INPUT...'}
            {state.stage !== 'concluded' && (
              <span className={`ml-1 text-[var(--color-cyan)] ${showCursor ? 'opacity-100' : 'opacity-0'}`}>▋</span>
            )}
          </span>
        </div>

        <div className="flex items-center gap-4">
          <div 
            className="flex items-center gap-2 rounded px-[16px] py-[6px] border"
            style={{
              borderColor: stageColor.replace(')', ', 0.3)').replace('var(', 'rgba('), // rough trick, we'll use inline standard
              background: `color-mix(in srgb, ${stageColor} 10%, transparent)`,
              border: `1px solid color-mix(in srgb, ${stageColor} 30%, transparent)`
            }}
          >
            <div 
              className="h-2 w-2 rounded-full" 
              style={{ 
                backgroundColor: stageColor,
                boxShadow: state.stage !== 'idle' ? `0 0 8px ${stageColor}` : 'none',
                animation: state.stage !== 'concluded' && state.stage !== 'idle' ? 'pulse-ring 1.5s infinite' : 'none'
              }} 
            />
            <span 
              className="text-[12px] uppercase tracking-[0.15em]"
              style={{ fontFamily: 'var(--font-mono)', color: stageColor }}
            >
              ⬡ {state.stage.toUpperCase()}
            </span>
          </div>
          <div className="mx-2 h-[20px] w-[1px] bg-[rgba(255,255,255,0.1)]" />
          <button 
            onClick={handleReset}
            className="text-[var(--color-text-muted)] transition-colors hover:text-[var(--color-crimson)]"
          >
            <X className="h-[20px] w-[20px]" />
          </button>
        </div>
      </header>

      {/* BODY */}
      <div className="grid h-screen w-full pt-[64px] pb-[48px] md:grid-cols-[240px_1fr] lg:grid-cols-[320px_1fr]">
        <div className="sticky top-[64px] hidden h-[calc(100vh-112px)] overflow-hidden md:block">
          <TerminalSidebar state={state} />
        </div>
        <div className="h-full min-h-0 overflow-y-auto">
          <InvestigationFeed state={state} onReset={handleReset} />
        </div>
      </div>

      <StatusBar />
    </div>
  );
}
