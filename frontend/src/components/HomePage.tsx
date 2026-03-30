import { motion } from 'framer-motion';
import { Activity, Brain, ShieldAlert } from 'lucide-react';
import EvilEye from './EvilEye';

interface HomePageProps {
  onStartInvestigation: () => void;
}

export default function HomePage({ onStartInvestigation }: HomePageProps) {
  return (
    <div className="relative w-full h-screen overflow-hidden">
      {/* ── EvilEye Background Layer ── */}
      <motion.div
        className="fixed inset-0 z-0"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 1.5, ease: 'easeOut' }}
      >
        <EvilEye
          eyeColor="#5c3870"
          intensity={1.5}
          pupilSize={0.6}
          irisWidth={0.25}
          glowIntensity={0.35}
          scale={0.8}
          noiseScale={1}
          pupilFollow={1}
          flameSpeed={1}
          backgroundColor="#060010"
        />
      </motion.div>

      {/* ── Content Layer ── */}
      <div className="relative z-10 flex flex-col items-center justify-center h-full px-6">
        {/* Title */}
        <div className="relative">
          <div 
            className="absolute inset-0 z-[-1] pointer-events-none rounded-full" 
            style={{ background: 'radial-gradient(ellipse 70% 40% at center, rgba(6,0,16,0.85) 0%, transparent 100%)' }} 
          />
          <motion.h1
            className="font-display text-[clamp(80px,12vw,160px)] leading-none tracking-[0.3em] text-ghost-white text-chromatic select-none"
            initial={{ x: -15, opacity: 0, filter: 'blur(6px)' }}
            animate={{ x: 0, opacity: 1, filter: 'blur(0px)' }}
            transition={{
              duration: 0.7,
              ease: [0.25, 0.46, 0.45, 0.94],
              delay: 0.8,
            }}
            style={{
              textShadow: `-2px 0 #FF1744, 2px 0 #00F5FF, 0 0 40px rgba(0, 245, 255, 0.6), 0 0 80px rgba(139, 0, 255, 0.3)`,
            }}
          >
          SHADOWMIND
          </motion.h1>
        </div>

        {/* Subtitle */}
        <motion.p
          className="font-mono text-[clamp(16px,2.5vw,28px)] tracking-[2px] text-[#E8E8F0] mt-4 text-center px-8 py-3 rounded-full"
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 1.4 }}
          style={{ 
            textShadow: '0 0 20px rgba(0,245,255,0.5)',
            border: '1px solid rgba(0,245,255,0.3)',
            backdropFilter: 'blur(8px)',
          }}
        >
          AUTONOMOUS INTELLIGENCE. RELENTLESS TRUTH.
        </motion.p>

        {/* Animated Gradient Line */}
        <motion.div
          className="gradient-line w-64 max-w-[80vw] mt-6"
          initial={{ scaleX: 0, opacity: 0 }}
          animate={{ scaleX: 1, opacity: 1 }}
          transition={{ duration: 0.8, delay: 1.6 }}
        />

        {/* CTA Button */}
        <motion.button
          id="cta-initiate"
          onClick={onStartInvestigation}
          className="group relative mt-10 min-w-[320px] px-12 py-5 font-mono text-[18px] tracking-[0.25em]
                     bg-white/5 backdrop-blur-md border-2 border-electric-cyan
                     text-electric-cyan cursor-pointer
                     transition-shadow duration-300
                     focus:outline-none focus:ring-1 focus:ring-electric-cyan/50
                     overflow-hidden"
          style={{ boxShadow: '0 0 40px rgba(0,245,255,0.4), inset 0 0 20px rgba(0,245,255,0.05)' }}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 1.8 }}
          whileHover={{ scale: 1.03, backgroundColor: 'rgba(0,245,255,0.15)' }}
          whileTap={{ scale: 0.98 }}
        >
          {/* Scanline sweep on hover */}
          <span className="absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity duration-300">
            <span
              className="absolute inset-0 bg-gradient-to-r from-transparent via-electric-cyan/10 to-transparent"
              style={{ animation: 'scanline-sweep 2s linear infinite' }}
            />
          </span>
          <span className="relative z-10">[ INITIATE INVESTIGATION ]</span>
        </motion.button>

        {/* Status Indicators */}
        <motion.div
          className="fixed bottom-8 left-8 flex flex-col gap-2.5 font-mono text-[14px] tracking-wider opacity-100"
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 2.2, staggerChildren: 0.1 }}
        >
          <StatusIndicator
            icon={<Activity className="w-3 h-3" />}
            label="SYS ONLINE"
            color="emerald"
            delay={2.2}
          />
          <StatusIndicator
            icon={<Brain className="w-3 h-3" />}
            label="NEURAL NET ACTIVE"
            color="amber"
            delay={2.4}
          />
          <StatusIndicator
            icon={<ShieldAlert className="w-3 h-3" />}
            label="THREAT LEVEL: UNKNOWN"
            color="crimson"
            delay={2.6}
          />
        </motion.div>
      </div>

      {/* ── Subtle scanline overlay ── */}
      <div
        className="fixed inset-0 z-20 pointer-events-none opacity-[0.03]"
        style={{
          background: `repeating-linear-gradient(
            0deg,
            transparent,
            transparent 2px,
            rgba(0, 245, 255, 0.15) 2px,
            rgba(0, 245, 255, 0.15) 4px
          )`,
        }}
      />
    </div>
  );
}

/* ── Status Indicator Sub-Component ── */
function StatusIndicator({
  icon,
  label,
  color,
  delay,
}: {
  icon: React.ReactNode;
  label: string;
  color: 'emerald' | 'amber' | 'crimson';
  delay: number;
}) {
  const dotColors = {
    emerald: 'bg-emerald-400 shadow-[0_0_8px_rgba(52,211,153,0.6)]',
    amber: 'bg-amber-400 shadow-[0_0_8px_rgba(251,191,36,0.6)]',
    crimson: 'bg-crimson-burn shadow-[0_0_8px_rgba(255,23,68,0.6)]',
  };

  const textColors = {
    emerald: 'text-emerald-400',
    amber: 'text-amber-400',
    crimson: 'text-crimson-burn',
  };

  return (
    <motion.div
      className={`flex items-center gap-2.5 ${textColors[color]}`}
      initial={{ opacity: 0, x: -20 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.4, delay }}
    >
      <span className={`w-2.5 h-2.5 rounded-full animate-pulse-glow ${dotColors[color]}`} />
      <span className="flex items-center gap-1.5 font-medium">
        {icon}
        {label}
      </span>
    </motion.div>
  );
}
