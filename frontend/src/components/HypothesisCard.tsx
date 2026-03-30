import { motion } from 'framer-motion';
import type { Hypothesis } from '@/hooks/useInvestigation';
import type { PipelineStage } from '@/hooks/useInvestigation';
import { cn, formatConfidence } from '@/lib/utils';
import { ExternalLink, ShieldCheck, AlertTriangle } from 'lucide-react';

interface HypothesisCardProps {
  hypothesis: Hypothesis;
  index: number;
  currentStage: PipelineStage;
}

export default function HypothesisCard({
  hypothesis,
  index,
  currentStage,
}: HypothesisCardProps) {
  const isComplete = ['scoring', 'concluded'].includes(currentStage);
  const confidenceValue = hypothesis.confidence ?? hypothesis.plausibility_score;
  const isDemoted = isComplete && confidenceValue < 0.35;
  const isSurvivor = isComplete && confidenceValue >= 0.35;
  
  const displayId = hypothesis.id.replace('hyp_', 'H-').toUpperCase();

  const getBorderColor = () => {
    if (isDemoted) return 'rgba(255, 23, 68, 1)';
    if (isSurvivor) return 'rgba(139, 0, 255, 1)';
    return 'rgba(0, 245, 255, 1)';
  };

  const getBoxShadow = () => {
    if (isSurvivor) return '0 0 40px rgba(139, 0, 255, 0.3)';
    if (isDemoted) return 'none'; // demoted cards have no outer glow
    return '0 0 20px rgba(0, 245, 255, 0.1)';
  };

  // Gradient based on score: Crimson (0%) -> Amber (50%) -> Cyan (75%) -> Violet (100%)
  const meterGradient = 'linear-gradient(90deg, #FF1744 0%, #FFB300 50%, #00F5FF 75%, #8B00FF 100%)';

  return (
    <motion.div
      layout
      transition={{ type: 'spring', stiffness: 120, damping: 20 }}
      initial={{ opacity: 0, y: -20 }}
      animate={{ opacity: 1, y: 0 }}
      className={cn(
        "relative flex flex-col w-full min-h-[280px] p-8 rounded-sm transition-all duration-700 overflow-hidden group border",
        isDemoted ? "grayscale-[80%] opacity-60 border-crimson-burn/30 animate-scanline-red" : "border-electric-cyan/20"
      )}
      style={{
        background: 'linear-gradient(135deg, rgba(10,2,28,0.95), rgba(20,5,45,0.9))',
        borderLeft: `4px solid ${getBorderColor()}`,
        boxShadow: getBoxShadow(),
      }}
      whileHover={!isDemoted ? { y: -4, borderColor: 'rgba(0,245,255,0.5)' } : {}}
    >
      {/* ── Top Bar ── */}
      <div className="flex justify-between items-center mb-6">
        <div className="flex items-center gap-3">
          <span className="font-mono text-[14px] text-electric-cyan font-bold tracking-widest">{displayId}</span>
          {isSurvivor && (
            <span className="flex items-center gap-1.5 px-3 py-1 bg-toxic-violet/10 border border-toxic-violet/30 text-toxic-violet font-mono text-[11px] tracking-widest uppercase rounded-sm shadow-[0_0_10px_rgba(139,0,255,0.2)]">
              <ShieldCheck className="w-3.5 h-3.5" />
              HYPOTHESIS SUSTAINED
            </span>
          )}
          {isDemoted && (
            <span className="flex items-center gap-1.5 px-3 py-1 bg-crimson-burn/10 border border-crimson-burn/30 text-crimson-burn font-mono text-[11px] tracking-widest uppercase rounded-sm">
              <AlertTriangle className="w-3.5 h-3.5" />
              LOW CONFIDENCE
            </span>
          )}
        </div>
        
        {/* We place the confidence at the end of the top bar just as a visual balance, or can omit it and use the massive bar */}
        <span className="font-mono text-[12px] text-[#B0B0C8] tracking-widest">
           CONFIDENCE: {formatConfidence(confidenceValue)}
        </span>
      </div>

      {/* ── Title & Description ── */}
      <h2 className="font-display text-[32px] tracking-[0.05em] text-white leading-none mb-3">
        {hypothesis.title}
      </h2>
      <p className="font-ui text-[15px] text-ghost-dim leading-relaxed max-w-[800px] mb-8">
        {hypothesis.description}
      </p>

      {/* ── Evidence Row ── */}
      {hypothesis.evidence && hypothesis.evidence.length > 0 && (
        <div className="mb-8 overflow-hidden">
          <div className="flex items-center gap-2 mb-3">
            <h4 className="font-mono text-[12px] tracking-[0.2em] text-[#B0B0C8]">EVIDENCE TRAIL</h4>
            <div className="h-px flex-1 bg-gradient-to-r from-border-glass to-transparent" />
          </div>
          
          <div className="flex gap-4 overflow-x-auto pb-4 scrollbar-thin scrollbar-thumb-electric-cyan/20 scrollbar-track-transparent">
            {hypothesis.evidence.map((ev, i) => {
              let domain = 'source';
              try { domain = new URL(ev.source_url).hostname.replace('www.', ''); } catch { }
              
              return (
                <div key={`${ev.id || i}`} className="flex-shrink-0 w-[240px] flex flex-col p-4 border border-electric-cyan/15 bg-electric-cyan/[0.02] rounded-sm hover:border-electric-cyan/40 transition-colors">
                  <div className="flex gap-3 mb-2">
                    <div className="w-8 h-8 rounded-sm bg-dark-matter flex items-center justify-center border border-border-glass shrink-0 text-[10px] text-electric-cyan font-mono overflow-hidden">
                       🌐
                    </div>
                    <div className="flex flex-col overflow-hidden">
                      <span className="font-ui text-[13px] font-bold text-white truncate">{ev.text.slice(0, 30)}...</span>
                      <span className="font-mono text-[10px] text-electric-cyan/70 truncate">{domain}</span>
                    </div>
                  </div>
                  
                  <p className="font-ui text-[11px] text-ghost-dim/60 line-clamp-2 mt-auto mb-3">
                    "{ev.text}"
                  </p>

                  <a 
                    href={ev.source_url} 
                    target="_blank" 
                    rel="noopener noreferrer"
                    className="flex items-center justify-center gap-1.5 w-full py-1.5 bg-electric-cyan/5 border border-electric-cyan/20 text-electric-cyan font-mono text-[11px] font-bold uppercase tracking-wider hover:bg-electric-cyan/15 transition-colors mt-auto"
                  >
                    <ExternalLink className="w-3 h-3" />
                    VISIT
                  </a>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* ── Elimination Reason Block ── */}
      {isDemoted && hypothesis.elimination_reason && (
        <motion.div 
          initial={{ opacity: 0, height: 0 }}
          animate={{ opacity: 1, height: 'auto' }}
          transition={{ duration: 0.5, delay: 0.3 }}
          className="mb-8 p-5 border-l-4 border-crimson-burn/80 bg-gradient-to-r from-crimson-burn/[0.08] to-transparent relative overflow-hidden"
        >
          <div className="flex items-center gap-2 mb-2">
            <AlertTriangle className="w-4 h-4 text-crimson-burn animate-pulse" />
            <h4 className="font-mono text-[12px] tracking-[0.2em] text-crimson-burn font-bold shadow-crimson-burn/20 drop-shadow-md">
              LLM ELIMINATION REASONING
            </h4>
          </div>
          <p className="font-mono text-[13px] text-white/80 leading-relaxed font-semibold">
            {hypothesis.elimination_reason}
          </p>
        </motion.div>
      )}

      {/* ── Confidence Meter Section ── */}
      <div className="mt-auto">
        <div className="flex items-end justify-between mb-2">
          <span className="font-mono text-[12px] tracking-[0.2em] text-[#B0B0C8] mb-1">CONFIDENCE METER</span>
          <span className="font-display text-[72px] leading-none text-white">{formatConfidence(confidenceValue)}</span>
        </div>
        
        <div className="h-4 bg-dark-matter border border-border-glass rounded-full overflow-hidden">
          <motion.div 
            className="h-full rounded-full"
            style={{ backgroundImage: meterGradient }}
            initial={{ width: '0%' }}
            animate={{ width: `${confidenceValue * 100}%` }}
            transition={{ duration: 1.2, ease: 'easeOut', delay: index * 0.1 }}
          />
        </div>
      </div>
      
    </motion.div>
  );
}
