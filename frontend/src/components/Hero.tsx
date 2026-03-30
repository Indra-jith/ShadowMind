import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { ChevronDown } from 'lucide-react';
import EvilEye from './EvilEye';

interface HeroProps {
  onStartInvestigation: (query: string) => void;
}

interface NavbarProps {
  onStart: () => void;
  isValid: boolean;
}

const Navbar: React.FC<NavbarProps> = ({ onStart, isValid }) => {
  return (
    <nav 
      className="fixed top-0 z-50 flex w-full items-center justify-between px-4 md:px-[120px] h-[68px]"
      style={{
        background: 'rgba(6,0,16,0.55)',
        backdropFilter: 'blur(20px) saturate(180%)',
        borderBottom: '1px solid rgba(0,245,255,0.07)',
      }}
    >
      <div className="flex items-center">
        <div 
          className="flex items-center text-[#E8E8F0] tracking-[0.2em]"
          style={{ fontFamily: 'var(--font-display)', fontSize: '26px' }}
        >
          <span 
            className="mr-2 text-[#00F5FF]" 
            style={{ filter: 'drop-shadow(0 0 8px #00F5FF)' }}
          >
            ◈
          </span>
          SHADOWMIND
        </div>
        <div className="ml-12 hidden md:flex items-center gap-7">
          <div 
            onClick={() => { if (isValid) onStart(); }}
            className={`group flex items-center gap-[6px] transition-all duration-200 ${isValid ? 'cursor-pointer text-[#7070a0] hover:text-[#00F5FF]' : 'cursor-not-allowed opacity-40 text-[#7070a0]'}`}
            style={{ fontFamily: 'var(--font-mono)', fontSize: '13px', letterSpacing: '0.15em' }}
          >
            <span className={isValid ? "group-hover:drop-shadow-[0_0_12px_rgba(0,245,255,0.6)]" : ""}>INVESTIGATE</span>
            <ChevronDown className="h-[14px] w-[14px]" />
          </div>
        </div>
      </div>
      <div>
        <motion.button
          onClick={() => { if (isValid) onStart(); }}
          whileHover={isValid ? { scale: 1.04, backgroundColor: 'rgba(0,245,255,0.1)', boxShadow: '0 0 40px rgba(0,245,255,0.35)' } : {}}
          transition={{ type: 'spring', stiffness: 300, damping: 20 }}
          className={`relative overflow-hidden rounded-full border border-[rgba(0,245,255,0.5)] px-[28px] py-[10px] backdrop-blur-[8px] ${isValid ? 'bg-[rgba(6,0,16,0.85)] cursor-pointer' : 'bg-transparent opacity-40 cursor-not-allowed'}`}
          style={{ boxShadow: isValid ? '0 0 20px rgba(0,245,255,0.15)' : 'none' }}
        >
          <div 
            className="absolute left-1/2 top-0 h-[1px] w-[60%] -translate-x-1/2"
            style={{
              background: isValid ? 'linear-gradient(90deg, transparent, rgba(0,245,255,0.8), transparent)' : 'none',
              filter: 'blur(1px)'
            }}
          />
          <span 
            className="text-[#00F5FF]"
            style={{ fontFamily: 'var(--font-mono)', fontSize: '13px', letterSpacing: '0.2em' }}
          >
            [ INITIATE ]
          </span>
        </motion.button>
      </div>
    </nav>
  );
};

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

interface HeroContentProps {
  query: string;
  setQuery: (val: string) => void;
  onStart: () => void;
  isValid: boolean;
}

const HeroContent: React.FC<HeroContentProps> = ({ query, setQuery, onStart, isValid }) => {
  return (
    <div className="relative z-10 flex flex-col items-center text-center pt-[180px] md:pt-[260px] pb-[100px] gap-[36px] px-4">
      
      {/* Element 1: Badge Pill */}
      <motion.div
        initial={{ y: -20, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ delay: 0.8, duration: 0.5 }}
        className="flex items-center rounded-full border border-[rgba(0,245,255,0.18)] bg-[rgba(0,245,255,0.06)] px-[20px] py-[8px] gap-[10px] backdrop-blur-[12px]"
      >
        <div className="h-2 w-2 rounded-full bg-[#00FF88]" style={{ animation: 'pulse-ring 2s infinite' }} />
        <span className="text-[13px] text-[rgba(232,232,240,0.5)]" style={{ fontFamily: 'var(--font-mono)' }}>Neural systems active —</span>
        <span className="text-[13px] text-[#00F5FF]" style={{ fontFamily: 'var(--font-mono)', letterSpacing: '0.08em' }}>READY FOR INVESTIGATION</span>
      </motion.div>

      {/* Element 2: Main Heading */}
      <motion.div
        initial={{ y: -30, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ delay: 1, duration: 0.6 }}
        className="relative"
        style={{
          animation: 'glitch-entry 0.3s forwards 1.6s'
        }}
      >
        <div 
          className="absolute inset-0 -z-10 rounded-full"
          style={{ background: 'radial-gradient(ellipse 90% 120% at center, rgba(6,0,16,0.9) 0%, transparent 70%)' }}
        />
        <h1 
          className="text-[#E8E8F0]"
          style={{ 
            fontFamily: 'var(--font-display)', 
            fontSize: 'clamp(64px, 10vw, 120px)', 
            lineHeight: 0.92, 
            letterSpacing: '0.02em',
            textShadow: '-2px 0 rgba(255,23,68,0.55), 2px 0 rgba(0,245,255,0.55), 0 0 50px rgba(0,245,255,0.2), 0 0 100px rgba(139,0,255,0.1)'
          }}
        >
          AUTONOMOUS INTELLIGENCE.<br />
          RELENTLESS TRUTH.
        </h1>
      </motion.div>

      {/* Element 3: Subtitle */}
      <motion.p
        initial={{ y: 20, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ delay: 1.3 }}
        className="max-w-[580px] text-[16px] leading-[1.75] text-[rgba(232,232,240,0.65)]"
        style={{ fontFamily: 'var(--font-body)' }}
      >
        ShadowMind deploys autonomous neural agents that decompose your query, 
        generate competing hypotheses, retrieve live evidence, and converge on truth — 
        in seconds.
      </motion.p>

      {/* Element 4: Animated Divider */}
      <div className="relative flex items-center justify-center pt-2">
        <motion.div
          initial={{ scaleX: 0 }}
          animate={{ scaleX: 1 }}
          transition={{ delay: 1.5, duration: 0.7, ease: "easeOut" }}
          className="h-[1px] w-[300px] origin-center"
          style={{ background: 'linear-gradient(90deg, transparent, #00F5FF 40%, #8B00FF 60%, transparent)' }}
        />
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 2.2 }}
          className="absolute h-[6px] w-[6px] rounded-full bg-[#00F5FF]"
          style={{ 
            boxShadow: '0 0 8px #00F5FF',
            left: '50%',
            animation: 'data-travel 3s infinite linear' 
          }}
        />
      </div>

      {/* Element 5: Primary CTA & Input */}
      <motion.div
        initial={{ scale: 0.88, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        transition={{ delay: 1.7, type: 'spring', stiffness: 220, damping: 22 }}
        className="flex flex-col items-center w-full max-w-[600px] gap-[24px]"
      >
        {/* Terminal Input */}
        <div className="w-full relative group">
          <div className="absolute left-[20px] top-[14px] text-[var(--color-cyan)] text-[14px] font-bold">{'>'}</div>
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && isValid) onStart();
            }}
            placeholder="ENTER INVESTIGATION QUERY..."
            className="w-full bg-[rgba(0,245,255,0.03)] border border-[rgba(0,245,255,0.15)] rounded-[4px] py-[12px] pl-[40px] pr-[16px] text-[15px] text-[#E8E8F0] placeholder:text-[rgba(112,112,160,0.4)] tracking-[0.05em] focus:outline-none focus:border-[var(--color-cyan)] focus:bg-[rgba(0,245,255,0.08)] transition-all"
            style={{ fontFamily: 'var(--font-mono)' }}
          />
        </div>

        <motion.button
          onClick={() => { if (isValid) onStart(); }}
          whileHover={isValid ? { scale: 1.04 } : { scale: 1 }}
          className={`group relative rounded-full p-[2px] transition-all duration-300 ${isValid ? 'cursor-pointer' : 'cursor-not-allowed opacity-40 grayscale'}`}
          style={{
            background: isValid ? 'linear-gradient(135deg, rgba(0,245,255,0.6), rgba(139,0,255,0.6))' : 'rgba(255,255,255,0.1)',
            boxShadow: isValid ? '0 0 60px rgba(0,245,255,0.2), 0 0 120px rgba(139,0,255,0.1)' : 'none'
          }}
        >
          <div className={`relative overflow-hidden rounded-full px-[52px] py-[18px] backdrop-blur-[16px] transition-colors duration-200 ${isValid ? 'bg-[rgba(6,0,16,0.9)] group-hover:bg-[rgba(0,245,255,0.1)] group-hover:shadow-[0_0_80px_rgba(0,245,255,0.5)]' : 'bg-[rgba(0,0,0,0.5)]'}`}>
            <div 
              className="absolute left-1/2 top-0 h-[1px] w-[60%] -translate-x-1/2"
              style={{
                background: isValid ? 'linear-gradient(90deg, transparent, rgba(0,245,255,0.8), transparent)' : 'none',
                filter: 'blur(3px)'
              }}
            />
            <span 
              className="text-[#00F5FF]"
              style={{ fontFamily: 'var(--font-mono)', fontSize: '15px', letterSpacing: '0.3em' }}
            >
              [ BEGIN INVESTIGATION ]
            </span>
          </div>
        </motion.button>
      </motion.div>

      {/* Element 6: Secondary Link */}
      <motion.button
        onClick={() => window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' })}
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 1.9 }}
        className="group mt-2 text-[13px] text-[rgba(112,112,160,0.7)] transition-all duration-200 hover:text-[#8B00FF]"
        style={{ fontFamily: 'var(--font-mono)', letterSpacing: '0.2em' }}
      >
        <span className="group-hover:drop-shadow-[0_0_12px_rgba(139,0,255,0.5)]">VIEW INTELLIGENCE BRIEF  ↓</span>
      </motion.button>
    </div>
  );
};

export default function Hero({ onStartInvestigation }: HeroProps) {
  const [query, setQuery] = useState('');
  const isValid = query.trim().length > 0;

  const handleInitiate = () => {
    if (!isValid) return;
    onStartInvestigation(query.trim());
  };

  return (
    <div className="relative min-h-screen w-full overflow-hidden bg-[var(--color-void)]">
      {/* Layer 0: EvilEye WebGL */}
      <div className="fixed inset-0 z-0 pointer-events-none">
        <motion.div
           initial={{ opacity: 0 }}
           animate={{ opacity: 1 }}
           transition={{ duration: 1.5, ease: "easeOut" }}
           className="h-full w-full"
        >
          <EvilEye 
            eyeColor="#5e40b0" 
            intensity={1.2} 
            pupilSize={0.9} 
            irisWidth={0.3}
            glowIntensity={0.5} 
            scale={0.7} 
            noiseScale={1} 
            pupilFollow={1} 
            flameSpeed={0.9}
            backgroundColor="#060010" 
          />
        </motion.div>
      </div>

      {/* Layer 1: Radial Overlay */}
      <div 
        className="pointer-events-none fixed inset-0 z-[1]"
        style={{ background: 'radial-gradient(ellipse 80% 60% at 50% 50%, rgba(6,0,16,0.2) 0%, rgba(6,0,16,0.6) 60%, rgba(6,0,16,0.9) 100%)' }}
      />

      <Navbar onStart={handleInitiate} isValid={isValid} />
      <HeroContent query={query} setQuery={setQuery} onStart={handleInitiate} isValid={isValid} />
      <StatusBar />
    </div>
  );
}
