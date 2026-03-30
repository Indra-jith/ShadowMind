import { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useInvestigation } from '@/hooks/useInvestigation';
import TerminalSidebar from './TerminalSidebar';
import HypothesisCard from './HypothesisCard';
import VerdictBlock from './VerdictBlock';
import { Search, ArrowUp, Hexagon } from 'lucide-react';

interface DashboardProps {
  onBack: () => void;
}

export default function Dashboard({ onBack }: DashboardProps) {
  const { state, startInvestigation, resetInvestigation } = useInvestigation();
  const [query, setQuery] = useState('');
  const inputRef = useRef<HTMLInputElement>(null);
  const [showScrollTop, setShowScrollTop] = useState(false);

  const hasStarted = state.stage !== 'idle';
  const showVerdict = state.stage === 'concluded' && state.verdict !== null;

  useEffect(() => {
    const handleScroll = () => {
      setShowScrollTop(window.scrollY > 400);
    };
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  function scrollToTop() {
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (query.trim().length < 5) return;
    startInvestigation(query.trim());
  }

  function handleNewInvestigation() {
    resetInvestigation();
    setQuery('');
    scrollToTop();
    setTimeout(() => inputRef.current?.focus(), 100);
  }

  // Render the header
  const getStageLabel = (stage: string) => {
    switch (stage) {
      case 'idle': return 'SYSTEM IDLE';
      case 'scanning': return 'DECOMPOSING MYSTERY';
      case 'generating': return 'GENERATING HYPOTHESES';
      case 'retrieving': return 'RETRIEVING EVIDENCE';
      case 'scoring': return 'SCORING & ELIMINATING';
      case 'concluded': return 'INVESTIGATION COMPLETE';
      default: return 'PROCESSING';
    }
  };

  return (
    <div className="flex flex-col lg:flex-row min-h-screen bg-dark-matter text-ghost-white overflow-visible relative">
      {/* ── Left Sidebar (Sticky) ── */}
      <div className="lg:w-[300px] flex-shrink-0 lg:sticky lg:top-0 lg:h-screen lg:overflow-y-auto z-40 border-r border-border-cyan/30 bg-dark-matter">
        <TerminalSidebar logs={state.terminalLogs} currentStage={state.stage} hypotheses={state.hypotheses} />
      </div>

      {/* ── Main Content Area (Scrollable Feed) ── */}
      <main className="flex-1 flex flex-col min-h-screen relative overflow-visible z-10 w-full">
        {/* ── Header Bar ── */}
        <header className="sticky top-0 z-30 h-[72px] flex flex-wrap lg:flex-nowrap items-center justify-between px-6 border-b border-border-glass bg-dark-glass">
          <div className="flex items-center gap-4 whitespace-nowrap">
            <button 
              onClick={onBack}
              className="font-display text-[28px] tracking-[0.1em] text-white hover:text-electric-cyan flex items-center gap-2 cursor-pointer transition-colors"
            >
              <Hexagon className="w-5 h-5 text-electric-cyan fill-electric-cyan/20" />
              SHADOWMIND
            </button>
            <div className="hidden lg:block w-px h-6 bg-border-glass" />
          </div>

          <form onSubmit={handleSubmit} className="flex-1 max-w-xl mx-4">
            <div className="relative group">
              <input
                ref={inputRef}
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                readOnly={hasStarted}
                placeholder="ENTER INVESTIGATION TARGET..."
                className="w-full bg-transparent border-none text-center font-mono text-[18px] text-electric-cyan placeholder:text-ghost-faint/30 focus:outline-none tracking-widest disabled:opacity-50"
                autoFocus
              />
              <div className="absolute bottom-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-electric-cyan/50 to-transparent scale-x-0 group-focus-within:scale-x-100 transition-transform duration-500" />
            </div>
          </form>

          <div className="flex items-center gap-3 whitespace-nowrap">
            <div className="flex items-center justify-center gap-2 px-3 py-1.5 rounded-sm border border-electric-cyan/30 bg-electric-cyan/5 shadow-[0_0_15px_rgba(0,245,255,0.1)]">
              <span className="w-1.5 h-1.5 rounded-full bg-electric-cyan animate-pulse-glow" />
              <span className="font-mono text-[12px] tracking-[0.1em] text-electric-cyan">
                {getStageLabel(state.stage)}
              </span>
            </div>
          </div>
        </header>

        {/* ── Investigation Feed ── */}
        <div className="p-8 w-full max-w-5xl mx-auto flex flex-col gap-10 pb-32">
          {!hasStarted ? (
            <div className="flex flex-col items-center justify-center h-[60vh] opacity-30 text-center">
              <Search className="w-16 h-16 text-ghost-faint mb-4" />
              <p className="font-mono text-sm tracking-widest text-ghost-faint">SYSTEM STANDBY. WAITING FOR INPUT.</p>
            </div>
          ) : (
            <AnimatePresence>
              {/* During Scanning: Skeletons */}
              {state.stage === 'scanning' && (
                <motion.div 
                  initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                  className="w-full space-y-8"
                >
                  {[1,2,3].map(i => (
                    <div key={i} className="w-full h-[280px] skeleton-shimmer rounded-sm border border-border-glass" />
                  ))}
                </motion.div>
              )}

              {/* Layout transition wrapper for cards */}
              <motion.div layout className="flex flex-col gap-10 w-full">
                {['surviving', 'active', 'pending', 'eliminated'].map(statusGroup => {
                   const matched = state.hypotheses.filter(h => {
                     const isComplete = ['scoring', 'concluded'].includes(state.stage);
                     const conf = h.confidence ?? h.plausibility_score;
                     
                     if (isComplete) {
                       if (statusGroup === 'eliminated') return conf < 0.35;
                       if (statusGroup === 'surviving') return conf >= 0.35;
                       return false;
                     }
                     
                     // Fallback string matching during processing
                     return h.status === statusGroup || (!h.status && statusGroup === 'active');
                   });

                   if (statusGroup === 'surviving') {
                     matched.sort((a,b) => (b.confidence ?? b.plausibility_score) - (a.confidence ?? a.plausibility_score));
                   }

                   return matched.map((hyp, i) => (
                     <HypothesisCard
                       key={`${hyp.id}-${statusGroup}-${i}`}
                       hypothesis={hyp}
                       index={i}
                       currentStage={state.stage}
                     />
                   ));
                })}
              </motion.div>
            </AnimatePresence>
          )}

          {/* ── Appended Verdict Block ── */}
          <AnimatePresence>
            {showVerdict && state.verdict && (
              <VerdictBlock
                verdict={state.verdict}
                hypotheses={state.hypotheses}
                onNewInvestigation={handleNewInvestigation}
              />
            )}
          </AnimatePresence>
        </div>
      </main>

      {/* ── Scroll To Top FAB ── */}
      <AnimatePresence>
        {showScrollTop && (
          <motion.button
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.8 }}
            onClick={scrollToTop}
            className="fixed bottom-8 right-8 z-50 p-4 rounded-full bg-dark-elevated border border-border-glass shadow-[0_0_30px_rgba(0,0,0,0.5)] hover:border-electric-cyan hover:shadow-[0_0_20px_rgba(0,245,255,0.3)] text-ghost-faint hover:text-electric-cyan transition-all cursor-pointer"
          >
            <ArrowUp className="w-5 h-5" />
          </motion.button>
        )}
      </AnimatePresence>
    </div>
  );
}
