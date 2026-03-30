import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { BrainCircuit, ExternalLink, ShieldCheck } from 'lucide-react';
import { Hypothesis } from '../hooks/useInvestigation';

interface HypothesisCardProps {
  hypothesis: Hypothesis;
  index: number;
}

export default function HypothesisCard({ hypothesis, index }: HypothesisCardProps) {
  const { status, confidence, eliminationReason, title, body, evidence } = hypothesis;
  const isEliminated = status === 'eliminated';
  const isSurvivor = status === 'survivor';
  const isActive = status === 'active';
  
  // Theme lookups
  const getAccentConfig = () => {
    switch(status) {
      case 'pending': return { bg: 'rgba(112,112,160,0.3)', shadow: 'none' };
      case 'active': return { bg: '#00F5FF', shadow: '0 0 20px rgba(0,245,255,0.8)' };
      case 'survivor': return { bg: '#8B00FF', shadow: '0 0 20px rgba(139,0,255,0.6)' };
      case 'eliminated': return { bg: '#FF1744', shadow: '0 0 16px rgba(255,23,68,0.4)' };
      default: return { bg: 'transparent', shadow: 'none' };
    }
  };
  const accent = getAccentConfig();

  const getConfColor = (conf: number) => {
    if (conf >= 0.75) return '#00FF88';
    if (conf >= 0.50) return '#00F5FF';
    if (conf >= 0.35) return '#FFB300';
    return '#FF1744';
  };
  const confColor = getConfColor(confidence || 0);

  const getConfGradient = (conf: number) => {
    if (conf < 0.35) return 'linear-gradient(90deg, #FF1744, rgba(255,23,68,0.3))';
    if (conf < 0.75) return 'linear-gradient(90deg, #FFB300, #00F5FF)';
    return 'linear-gradient(90deg, #00F5FF, #8B00FF)';
  };

  const getStatusBadge = () => {
    if (isSurvivor) return { text: 'SUSTAINED', color: '#8B00FF', bg: 'rgba(139,0,255,0.1)', border: '1px solid rgba(139,0,255,0.3)' };
    if (isEliminated) return { text: 'ELIMINATED', color: '#FF1744', bg: 'rgba(255,23,68,0.1)', border: '1px solid rgba(255,23,68,0.3)' };
    if (isActive) return { text: 'ANALYZING', color: '#00F5FF', bg: 'rgba(0,245,255,0.1)', border: '1px solid rgba(0,245,255,0.3)' };
    return { text: 'PENDING', color: 'rgba(112,112,160,0.6)', bg: 'transparent', border: '1px solid rgba(112,112,160,0.3)' };
  };
  const badge = getStatusBadge();

  return (
    <div 
      className="relative w-full min-h-[320px] rounded-[8px] overflow-hidden"
      style={{
        background: 'linear-gradient(135deg, rgba(16,8,32,0.97) 0%, rgba(22,11,44,0.95) 100%)',
        border: isSurvivor ? '1px solid rgba(139,0,255,0.25)' : '1px solid rgba(0,245,255,0.1)',
        boxShadow: isSurvivor ? '0 0 40px rgba(139,0,255,0.15), 0 8px 32px rgba(139,0,255,0.08)' : 'none',
        filter: isEliminated ? 'grayscale(0.8)' : 'blur(0)',
        opacity: isEliminated ? 0.65 : 1,
        transition: 'filter 1s ease, opacity 0.8s ease',
        animation: `card-materialize 0.6s cubic-bezier(0.22, 1, 0.36, 1) ${index * 0.12}s forwards`
      }}
    >
      {/* Accent Bar */}
      <div 
        className="absolute left-0 top-0 h-full w-[4px]"
        style={{
          backgroundColor: accent.bg,
          boxShadow: accent.shadow,
          transition: 'all 0.4s ease'
        }}
      />

      {/* OVERLAYS FOR ELIMINATED */}
      {isEliminated && (
        <>
          <div 
            className="pointer-events-none absolute inset-0 z-10"
            style={{
              background: 'linear-gradient(transparent, rgba(255,23,68,0.06), transparent)',
              height: '40%',
              animation: 'scanline-red 1.5s ease forwards'
            }}
          />
          <div 
            className="pointer-events-none absolute inset-0 z-10"
            style={{
              background: 'repeating-linear-gradient(180deg, transparent, transparent 4px, rgba(255,23,68,0.02) 4px, rgba(255,23,68,0.02) 8px)'
            }}
          />
        </>
      )}

      {/* CARD HEADER ROW */}
      <div className="flex justify-between items-start pt-[24px] pb-[16px] pl-[32px] pr-[28px]">
        {/* Left Header */}
        <div>
          <div 
            className="inline-block rounded-[3px] border border-[rgba(0,245,255,0.15)] bg-[rgba(0,245,255,0.06)] px-[8px] py-[3px] text-[11px] tracking-[0.2em] text-[rgba(0,245,255,0.6)]"
            style={{ fontFamily: 'var(--font-mono)' }}
          >
            H-00{index + 1}
          </div>
          <h2 
            className="mt-[10px] max-w-[600px] text-[#E8E8F0] tracking-[0.02em] leading-none"
            style={{ fontFamily: 'var(--font-display)', fontSize: 'clamp(24px, 3vw, 36px)' }}
          >
            {title}
          </h2>
        </div>

        {/* Right Header: Confidence */}
        <div className="flex flex-col items-end">
          <div className="flex items-center gap-3">
            {isSurvivor && (
              <div 
                className="flex items-center gap-1.5 px-[10px] py-[4px] rounded-[3px]"
                style={{ background: badge.bg, border: badge.border, color: badge.color, fontFamily: 'var(--font-mono)', fontSize: '11px', letterSpacing: '0.15em' }}
              >
                <ShieldCheck className="w-[12px] h-[12px]" />
                HYPOTHESIS SUSTAINED
              </div>
            )}
            {!isSurvivor && (
              <div 
                className="px-[10px] py-[4px] rounded-[3px]"
                style={{ background: badge.bg, border: badge.border, color: badge.color, fontFamily: 'var(--font-mono)', fontSize: '11px', letterSpacing: '0.15em' }}
              >
                {badge.text}
              </div>
            )}
          </div>
          <div className="mt-2 flex items-baseline">
            <span 
              className="leading-none transition-colors duration-[1.4s]"
              style={{ fontFamily: 'var(--font-display)', fontSize: '64px', color: confColor }}
            >
              {Math.round(confidence * 100)}
            </span>
            <span 
              className="leading-none transition-colors duration-[1.4s]"
              style={{ fontFamily: 'var(--font-display)', fontSize: '36px', color: confColor, opacity: 0.5 }}
            >
              %
            </span>
          </div>
          <span 
            className="-mt-1 text-[10px] tracking-[0.2em] text-[rgba(112,112,160,0.6)]"
            style={{ fontFamily: 'var(--font-mono)' }}
          >
            CONFIDENCE
          </span>
        </div>
      </div>

      {/* CONFIDENCE METER BAR */}
      <div className="ml-[32px] h-[3px] w-[calc(100%-32px)] rounded-[1.5px] bg-[rgba(255,255,255,0.06)] overflow-hidden">
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${confidence * 100}%` }}
          transition={{ duration: 1.4, ease: [0.22, 1, 0.36, 1], delay: 0.3 }}
          className="h-full"
          style={{ background: getConfGradient(confidence) }}
        />
      </div>

      {/* ELIMINATION REASON AFTER FAILURE */}
      <AnimatePresence>
        {isEliminated && eliminationReason && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            className="overflow-hidden"
          >
            <div 
              className="mt-[16px] mx-[28px] ml-[32px] rounded-[4px] border border-[rgba(255,23,68,0.15)] bg-[rgba(255,23,68,0.04)] p-[16px]"
            >
              <div className="flex items-center gap-2">
                <BrainCircuit className="h-[16px] w-[16px] text-crimson" style={{ color: '#FF1744' }} />
                <span 
                  className="text-[11px] tracking-[0.15em] text-[#FF1744]"
                  style={{ fontFamily: 'var(--font-mono)' }}
                >
                  LLM ELIMINATION REASONING
                </span>
              </div>
              <p 
                className="mt-[8px] text-[13px] leading-[1.6] text-[rgba(232,232,240,0.55)]"
                style={{ fontFamily: 'var(--font-body)' }}
              >
                {eliminationReason}
              </p>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* HYPOTHESIS BODY */}
      <div 
        className="pl-[32px] pr-[28px] py-[16px] max-w-[800px] text-[15px] leading-[1.75] text-[rgba(232,232,240,0.75)]"
        style={{ fontFamily: 'var(--font-body)' }}
      >
        {body}
      </div>

      {/* EVIDENCE TRAIL SECTION */}
      {evidence && evidence.length > 0 && (
        <div className="ml-[32px] mt-2 pb-6">
          <div className="grid grid-cols-[1fr_auto_1fr] items-center gap-[12px] px-[28px]">
            <div className="h-[1px] w-full bg-[rgba(0,245,255,0.12)]" />
            <span 
              className="text-[11px] tracking-[0.25em] text-[rgba(0,245,255,0.4)]"
              style={{ fontFamily: 'var(--font-mono)' }}
            >
              ─── EVIDENCE TRAIL ──
            </span>
            <div className="h-[1px] w-full bg-[rgba(0,245,255,0.12)]" />
          </div>

          <div 
            className="mt-4 flex gap-[12px] overflow-x-auto px-[28px] pb-[8px] scroll-smooth"
            style={{ scrollSnapType: 'x mandatory' }}
          >
            {evidence.map((ev, i) => (
              <div
                key={i}
                className="group flex w-[220px] flex-shrink-0 flex-col rounded-[6px] border border-[rgba(0,245,255,0.1)] bg-[rgba(0,0,0,0.3)] p-[12px] backdrop-blur-[4px] transition-all duration-200 hover:-translate-y-[2px] hover:border-[rgba(0,245,255,0.3)]"
                style={{ scrollSnapAlign: 'start' }}
              >
                <div className="flex items-center gap-2 mb-2">
                  {ev.favicon ? (
                    <img src={ev.favicon} alt="" className="h-4 w-4 rounded-sm" />
                  ) : (
                    <span className="text-[16px]">🌐</span>
                  )}
                  <span 
                    className="truncate text-[11px] text-[rgba(0,245,255,0.6)]"
                    style={{ fontFamily: 'var(--font-mono)' }}
                  >
                    {ev.domain}
                  </span>
                </div>
                
                <p 
                  className="mb-3 text-[12px] leading-[1.5] text-[rgba(232,232,240,0.6)] flex-1 overflow-hidden"
                  style={{ 
                    fontFamily: 'var(--font-body)',
                    display: '-webkit-box',
                    WebkitLineClamp: 3,
                    WebkitBoxOrient: 'vertical'
                  }}
                >
                  "{ev.excerpt}"
                </p>

                <a 
                  href={ev.url} 
                  target="_blank" 
                  rel="noopener noreferrer"
                  className="flex items-center gap-[4px] text-[11px] tracking-[0.1em] text-[#00F5FF] transition-all hover:drop-shadow-[0_0_8px_rgba(0,245,255,0.6)]"
                  style={{ fontFamily: 'var(--font-mono)' }}
                >
                  <ExternalLink className="h-[12px] w-[12px]" />
                  ↗ OPEN SOURCE
                </a>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
