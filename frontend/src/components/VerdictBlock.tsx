import React, { useEffect, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { ArrowUp, Download, RefreshCw } from 'lucide-react';
import { InvestigationState } from '../hooks/useInvestigation';

interface VerdictBlockProps {
  verdict: InvestigationState['verdict'];
  onReset: () => void;
}

const CircularProgress: React.FC<{ value: number }> = ({ value }) => {
  const [offset, setOffset] = useState(300);
  
  useEffect(() => {
    // 300 is our assumed dasharray approx circumference for r=48
    const toOffset = 300 - (300 * value);
    const timer = setTimeout(() => setOffset(toOffset), 100);
    return () => clearTimeout(timer);
  }, [value]);

  return (
    <div className="relative flex items-center justify-center">
      <svg className="w-[120px] h-[120px] transform -rotate-90">
        <circle 
          cx="60" cy="60" r="48" 
          fill="none" 
          stroke="rgba(255,255,255,0.06)" 
          strokeWidth="6" 
        />
        <circle 
          cx="60" cy="60" r="48" 
          fill="none" 
          stroke="#8B00FF" 
          strokeWidth="6"
          strokeLinecap="round"
          strokeDasharray="300"
          strokeDashoffset={offset}
          style={{ transition: 'stroke-dashoffset 1.8s cubic-bezier(0.22, 1, 0.36, 1)' }}
        />
      </svg>
      <div 
        className="absolute inset-0 flex items-center justify-center text-[#8B00FF]"
        style={{ fontFamily: 'var(--font-display)', fontSize: '48px', textShadow: '0 0 8px rgba(139,0,255,0.8)' }}
      >
        {Math.round(value * 100)}<span className="text-[24px] opacity-50 ml-1">%</span>
      </div>
    </div>
  );
};

export default function VerdictBlock({ verdict, onReset }: VerdictBlockProps) {
  const [showFab, setShowFab] = useState(false);

  useEffect(() => {
    const handleScroll = () => {
      // Show FAB if scrolled down even a little
      if (window.scrollY > 200) setShowFab(true);
      else setShowFab(false);
    };
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  const handleExportDossier = () => {
    if (!verdict) return;
    const content = `══════════════════════════════════════
SHADOWMIND INTELLIGENCE DOSSIER
══════════════════════════════════════
GENERATED: ${new Date().toLocaleString()}

VERDICT FILED: ${verdict.status || 'INCONCLUSIVE'}
CONFIDENCE: ${Math.round((verdict.confidence || 0) * 100)}%

--- NEURAL CONCLUSION ---
${verdict.narrative || 'None'}

--- LIMITATIONS ---
${verdict.caveats?.map(c => `• ${c}`).join('\n') || 'None'}

--- INTELLIGENCE SOURCES ---
${verdict.sources?.map(s => `• ${s}`).join('\n') || 'None'}
`;

    const blob = new Blob([content], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `shadowmind_dossier_${new Date().getTime()}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  if (!verdict) return null;

  return (
    <>
      <div 
        className="relative mt-[48px] overflow-hidden rounded-[8px] p-[28px] md:p-[48px_40px]"
        style={{
          background: 'linear-gradient(135deg, rgba(13,0,32,0.98), rgba(26,0,64,0.95))',
          border: '2px solid rgba(139,0,255,0.35)',
          boxShadow: '0 0 80px rgba(139,0,255,0.2), 0 0 160px rgba(139,0,255,0.08), inset 0 0 60px rgba(139,0,255,0.03)'
        }}
      >
        {/* Background Texture & Scanline */}
        <div 
          className="pointer-events-none absolute inset-0"
          style={{
            background: 'repeating-linear-gradient(0deg, transparent, transparent 30px, rgba(139,0,255,0.01) 30px, rgba(139,0,255,0.01) 31px)'
          }}
        />
        <div 
          className="pointer-events-none absolute inset-0 bg-gradient-to-b from-transparent via-[rgba(139,0,255,0.2)] to-transparent"
          style={{ height: '40%', animation: 'scanline-sweep 2s ease forwards' }}
        />

        {/* TOP LABEL */}
        <div className="relative flex items-center mb-[16px]">
          <div className="flex-1 h-[1px] bg-[rgba(139,0,255,0.2)]" />
          <span 
            className="px-4 text-[13px] tracking-[0.4em] text-[rgba(139,0,255,0.6)]"
            style={{ fontFamily: 'var(--font-mono)' }}
          >
            ══ VERDICT FILED ══
          </span>
          <div className="flex-1 h-[1px] bg-[rgba(139,0,255,0.2)]" />
        </div>

        {/* VERDICT AND ARC */}
        <div className="relative flex flex-col md:flex-row items-center justify-between gap-8 mb-[32px]">
          <h2 
            className="text-[#8B00FF]"
            style={{
              fontFamily: 'var(--font-display)',
              fontSize: 'clamp(52px, 7vw, 96px)',
              letterSpacing: '0.05em',
              textShadow: '0 0 40px rgba(139,0,255,0.6), 0 0 80px rgba(139,0,255,0.3)'
            }}
          >
            {verdict.status || 'INCONCLUSIVE'}
          </h2>
          <CircularProgress value={verdict.confidence} />
        </div>

        {/* FINAL NARRATIVE */}
        <div className="relative mb-[32px]">
          <span 
            className="block mb-4 text-[11px] tracking-[0.25em] text-[rgba(139,0,255,0.6)]"
            style={{ fontFamily: 'var(--font-mono)' }}
          >
            ─── NEURAL CONCLUSION ──
          </span>
          <p 
            className="max-w-[800px] text-[16px] leading-[1.8] text-[rgba(232,232,240,0.8)]"
            style={{ fontFamily: 'var(--font-body)' }}
          >
            {verdict.narrative}
          </p>
        </div>

        {/* CAVEATS */}
        {verdict.caveats && verdict.caveats.length > 0 && (
          <div className="relative mb-[32px]">
            <span 
              className="block mb-3 text-[12px] tracking-[0.2em] text-[#FFB300]"
              style={{ fontFamily: 'var(--font-mono)' }}
            >
              ⚠ LIMITATIONS
            </span>
            <ul className="flex flex-col gap-2">
              {verdict.caveats.map((cav, i) => (
                <li key={i} className="flex items-start gap-2">
                  <div className="mt-[8px] h-1.5 w-1.5 rounded-full bg-[#FFB300] shadow-[0_0_8px_#FFB300]" />
                  <span className="text-[14px] text-[rgba(232,232,240,0.6)]" style={{ fontFamily: 'var(--font-body)' }}>{cav}</span>
                </li>
              ))}
            </ul>
          </div>
        )}

        {/* SOURCES */}
        {verdict.sources && verdict.sources.length > 0 && (
          <div className="relative mb-[32px]">
            <span 
              className="block mb-3 text-[12px] tracking-[0.15em] text-[rgba(0,245,255,0.6)]"
              style={{ fontFamily: 'var(--font-mono)' }}
            >
              ◎ INTELLIGENCE SOURCES
            </span>
            <div className="flex flex-wrap gap-[8px]">
              {verdict.sources.map((src, i) => (
                <div 
                  key={i}
                  className="rounded-[3px] border border-[rgba(0,245,255,0.1)] bg-[rgba(0,245,255,0.04)] px-[10px] py-[4px] text-[12px] text-[rgba(0,245,255,0.6)]"
                  style={{ fontFamily: 'var(--font-mono)' }}
                >
                  {src}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* BUITONS */}
        <div className="relative mt-[32px] flex flex-wrap gap-[16px]">
          <button
            onClick={handleExportDossier}
            className="group relative rounded-full p-[2px]"
            style={{
              background: 'linear-gradient(135deg, rgba(139,0,255,0.6), rgba(0,245,255,0.6))',
              boxShadow: '0 0 60px rgba(139,0,255,0.2)'
            }}
          >
            <div className="flex items-center gap-2 overflow-hidden rounded-full bg-[rgba(6,0,16,0.9)] px-[24px] py-[12px] backdrop-blur-[16px] transition-colors duration-200 group-hover:bg-[rgba(139,0,255,0.1)] group-hover:shadow-[0_0_80px_rgba(139,0,255,0.5)]">
               <Download className="h-[14px] w-[14px] text-[#8B00FF]" />
               <span 
                className="text-[#8B00FF]"
                style={{ fontFamily: 'var(--font-mono)', fontSize: '12px', letterSpacing: '0.2em' }}
              >
                [ EXPORT DOSSIER ]
              </span>
            </div>
          </button>

          <button
            onClick={onReset}
            className="group flex items-center gap-2 rounded-full border border-[rgba(255,255,255,0.15)] bg-transparent px-[24px] py-[14px] transition-colors hover:border-[rgba(232,232,240,0.6)] hover:bg-[rgba(255,255,255,0.05)]"
          >
             <RefreshCw className="h-[14px] w-[14px] text-[rgba(232,232,240,0.7)] group-hover:text-[#E8E8F0]" />
             <span 
              className="text-[rgba(232,232,240,0.7)] group-hover:text-[#E8E8F0]"
              style={{ fontFamily: 'var(--font-mono)', fontSize: '12px', letterSpacing: '0.2em' }}
            >
              [ NEW INVESTIGATION ]
            </span>
          </button>
        </div>
      </div>

      {/* FAB */}
      <AnimatePresence>
        {showFab && (
          <motion.button
            initial={{ opacity: 0, scale: 0 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0 }}
            whileHover={{ scale: 1.1, boxShadow: '0 0 40px rgba(0,245,255,0.5)' }}
            onClick={() => window.scrollTo({ top: 0, behavior: 'smooth' })}
            className="fixed bottom-[64px] right-[24px] z-[100] flex h-[44px] w-[44px] items-center justify-center rounded-full border border-[rgba(0,245,255,0.3)] bg-[rgba(6,0,16,0.9)]"
            style={{ boxShadow: '0 0 20px rgba(0,245,255,0.2)' }}
          >
            <ArrowUp className="h-[20px] w-[20px] text-[var(--color-cyan)]" />
          </motion.button>
        )}
      </AnimatePresence>
    </>
  );
}
