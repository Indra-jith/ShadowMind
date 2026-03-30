import { motion } from 'framer-motion';
import type { PipelineStage } from '@/hooks/useInvestigation';
import { cn } from '@/lib/utils';
import { Check, Loader2 } from 'lucide-react';

interface PipelineNodeProps {
  label: string;
  stage: string;
  currentStage: PipelineStage;
  index: number;
}

const STAGE_ORDER: PipelineStage[] = [
  'scanning',
  'generating',
  'retrieving',
  'scoring',
  'concluded',
];

const STAGE_DESCRIPTIONS: Record<string, string> = {
  'scanning': 'Breaking query into investigation vectors...',
  'generating': 'Formulating viable hypotheses...',
  'retrieving': 'Extracting source evidence from neural net...',
  'scoring': 'Evaluating confidence and eliminating theories...',
  'concluded': 'Synthesizing final verdict from survivors.',
};

function getNodeStatus(
  nodeStage: string,
  currentStage: PipelineStage
): 'pending' | 'active' | 'complete' | 'failed' {
  const nodeIndex = STAGE_ORDER.indexOf(nodeStage as PipelineStage);
  const currentIndex = STAGE_ORDER.indexOf(currentStage);

  if (currentStage === 'idle') return 'pending';
  if (nodeIndex < currentIndex) return 'complete';
  if (nodeIndex === currentIndex) return 'active';
  return 'pending';
}

export default function PipelineNode({
  label,
  stage,
  currentStage,
  index,
}: PipelineNodeProps) {
  const status = getNodeStatus(stage, currentStage);
  const desc = STAGE_DESCRIPTIONS[stage];

  return (
    <motion.div
      className={cn(
        "flex flex-col min-h-[90px] p-4 transition-all duration-500 border-l-4 mb-2 bg-dark-elevated/40",
        status === 'active' && 'border-electric-cyan shadow-[-4px_0_20px_rgba(0,245,255,0.3)] bg-electric-cyan/5',
        status === 'complete' && 'border-toxic-violet',
        status === 'pending' && 'border-ghost-faint/10',
        status === 'failed' && 'border-crimson-burn'
      )}
      initial={{ opacity: 0, x: -20 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ delay: index * 0.1, duration: 0.3 }}
    >
      <div className="flex justify-between items-start mb-2">
        <h4 className={cn(
          "font-mono text-[15px] font-bold tracking-widest",
          status === 'active' ? 'text-electric-cyan' : status === 'complete' ? 'text-ghost-white' : 'text-ghost-faint'
        )}>
          ① {label}
        </h4>
        
        {status === 'complete' && (
          <Check className="w-5 h-5 text-toxic-violet" strokeWidth={3} />
        )}
        {status === 'active' && (
          <Loader2 className="w-4 h-4 text-electric-cyan animate-spin" />
        )}
      </div>

      <div className="flex justify-between items-end mt-auto">
        <p className="font-ui text-[13px] text-[#B0B0C8] leading-tight max-w-[80%] pr-4">
          {desc}
        </p>

        {status === 'active' && (
          <span className="font-mono text-[10px] tracking-widest text-electric-cyan animate-pulse">
            ACTIVE ●
          </span>
        )}
      </div>
    </motion.div>
  );
}
