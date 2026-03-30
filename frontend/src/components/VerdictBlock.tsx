import { motion } from 'framer-motion';
import type { Conclusion } from '@/hooks/useInvestigation';
import type { Hypothesis } from '@/hooks/useInvestigation';
import { formatConfidence } from '@/lib/utils';
import { FileDown, RotateCcw, ShieldCheck, AlertCircle, ExternalLink } from 'lucide-react';

interface VerdictBlockProps {
  verdict: Conclusion;
  hypotheses: Hypothesis[];
  onNewInvestigation: () => void;
}

export default function VerdictBlock({
  verdict,
  hypotheses,
  onNewInvestigation,
}: VerdictBlockProps) {
  const survivor = hypotheses.find(
    (h) => h.id === verdict.surviving_hypothesis
  );
  const isResolved = verdict.overall_confidence >= 0.5;

  return (
    <motion.div
      className="relative w-full rounded-sm border-2 overflow-hidden mt-6"
      style={{
        background: 'linear-gradient(135deg, #0D0020 0%, #1A0040 100%)',
        borderColor: '#8B00FF',
        boxShadow: '0 0 60px rgba(139,0,255,0.4)'
      }}
      initial={{ y: 50, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ type: 'spring', damping: 25, stiffness: 200, duration: 0.8 }}
    >
      <div className="flex items-center justify-center p-3 border-b border-toxic-violet/30 bg-toxic-violet/10">
         <span className="font-mono text-[12px] tracking-[0.2em] text-toxic-violet uppercase font-bold">
            ══════════════ VERDICT FILED ══════════════
         </span>
      </div>

      <div className="p-10">
        {/* ── Verdict Title ── */}
        <div className="text-center mb-10">
          <div className="flex items-center justify-center gap-4 mb-4">
            {isResolved ? (
              <ShieldCheck className="w-10 h-10 text-electric-cyan" />
            ) : (
              <AlertCircle className="w-10 h-10 text-amber-400" />
            )}
            <h2
              className="font-display text-[64px] tracking-[0.1em] leading-none text-white"
              style={{
                textShadow: isResolved ? '0 0 40px rgba(139, 0, 255, 0.8)' : '0 0 40px rgba(255, 167, 38, 0.6)',
              }}
            >
              {isResolved ? 'CASE RESOLVED' : 'INCONCLUSIVE'}
            </h2>
          </div>

          <div className="flex items-center justify-center gap-4 font-mono text-[14px] tracking-wider text-[#B0B0C8]">
            <span>CONFIDENCE: {verdict.confidence_label.toUpperCase()}</span>
            <span className="text-ghost-faint/30">|</span>
            <span>
              SURVIVORS:{' '}
              {hypotheses.filter((h) => (h.confidence ?? h.plausibility_score) >= 0.35).length}/
              {hypotheses.length}
            </span>
          </div>
        </div>

        {/* ── Summary & Details ── */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-10 mb-10">
           {/* Left Column: Summary */}
           <div className="flex flex-col gap-4">
              <h4 className="font-mono text-[14px] tracking-[0.2em] text-electric-cyan uppercase">
                Investigation Findings
              </h4>
              <p className="font-ui text-[16px] leading-relaxed text-ghost-white">
                {verdict.summary}
              </p>
              {survivor && (
                 <div className="mt-4 p-5 rounded-sm bg-black/40 border border-border-cyan/30">
                   <h5 className="font-mono text-[12px] text-electric-cyan tracking-widest mb-2">PRIMARY HYPOTHESIS OVERVIEW</h5>
                   <p className="font-ui text-[14px] text-[#B0B0C8] line-clamp-3">{survivor.description}</p>
                 </div>
              )}
           </div>

           {/* Right Column: Caveats & Sources */}
           <div className="flex flex-col gap-6">
              {verdict.caveats.length > 0 && (
                <div className="p-5 rounded-sm bg-amber-400/5 border border-amber-400/15">
                  <h4 className="font-mono text-[12px] tracking-[0.2em] text-amber-400/70 mb-3 uppercase">Caveats</h4>
                  <ul className="space-y-2">
                    {verdict.caveats.map((caveat, i) => (
                      <li key={i} className="font-mono text-[13px] text-ghost-dim/70 flex items-start gap-3">
                        <span className="text-amber-400 mt-0.5">⚠</span>
                        {caveat}
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {verdict.all_sources.length > 0 && (
                <div>
                  <h4 className="font-mono text-[12px] tracking-[0.2em] text-ghost-faint/50 mb-3 uppercase">
                    Cited Sources ({verdict.all_sources.length})
                  </h4>
                  <div className="flex flex-wrap gap-2">
                    {verdict.all_sources.map((url, i) => {
                      let domain = 'source';
                      try { domain = new URL(url).hostname.replace('www.', ''); } catch { }
                      return (
                        <a
                          key={i}
                          href={url}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="flex items-center gap-1.5 px-3 py-1.5 rounded-sm bg-ghost-faint/5 border border-border-glass
                                     font-mono text-[11px] text-[#B0B0C8] hover:text-electric-cyan hover:border-electric-cyan
                                     transition-all duration-200"
                        >
                          <ExternalLink className="w-3 h-3" />
                          {domain}
                        </a>
                      );
                    })}
                  </div>
                </div>
              )}
           </div>
        </div>

        {/* ── Final Meter & Actions ── */}
        <div className="border-t border-toxic-violet/20 pt-8">
           <div className="flex items-center justify-between mb-4">
             <span className="font-mono text-[14px] tracking-widest text-[#B0B0C8] uppercase">
                FINAL PROBABILITY SCORES
             </span>
             <span className="font-mono text-[32px] font-bold text-electric-cyan">
               {formatConfidence(verdict.overall_confidence)}
             </span>
           </div>
           
           <div className="h-4 rounded-full bg-dark-matter/50 overflow-hidden border border-border-glass mb-10 w-full relative">
             <motion.div
               className="h-full rounded-full"
               style={{ background: 'linear-gradient(90deg, #00F5FF, #8B00FF)', boxShadow: '0 0 15px rgba(139, 0, 255, 0.5)' }}
               initial={{ width: 0 }}
               animate={{ width: `${verdict.overall_confidence * 100}%` }}
               transition={{ duration: 1.5, delay: 0.6, ease: 'easeOut' }}
             />
           </div>

           <div className="flex flex-col sm:flex-row gap-4">
             <button className="flex-1 flex items-center justify-center gap-3 px-8 py-4 font-mono text-[14px] tracking-[0.2em] bg-toxic-violet/10 border border-toxic-violet/30 text-toxic-violet hover:bg-toxic-violet/20 hover:border-toxic-violet transition-all duration-300 cursor-pointer">
               <FileDown className="w-5 h-5" />
               EXPORT FULL DOSSIER
             </button>

             <button onClick={onNewInvestigation} className="flex-1 flex items-center justify-center gap-3 px-8 py-4 font-mono text-[14px] tracking-[0.2em] bg-electric-cyan/10 border border-electric-cyan/30 text-electric-cyan hover:bg-electric-cyan/20 hover:border-electric-cyan transition-all duration-300 cursor-pointer shadow-[0_0_20px_rgba(0,245,255,0.1)] hover:shadow-[0_0_30px_rgba(0,245,255,0.3)]">
               <RotateCcw className="w-5 h-5" />
               NEW INVESTIGATION
             </button>
           </div>
        </div>

      </div>
    </motion.div>
  );
}
