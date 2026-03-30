import { useState } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import Hero from '@/components/Hero';
import Dashboard from '@/components/Dashboard';

type View = 'home' | 'dashboard';

export default function App() {
  const [view, setView] = useState<View>('home');
  const [query, setQuery] = useState('');

  return (
    <AnimatePresence mode="wait">
      {view === 'home' ? (
        <motion.div
          key="home"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.5 }}
        >
          <Hero onStartInvestigation={(q) => { setQuery(q); setView('dashboard'); }} />
        </motion.div>
      ) : (
        <motion.div
          key="dashboard"
          className="h-screen"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.5 }}
        >
          <Dashboard initialQuery={query} onBack={() => setView('home')} />
        </motion.div>
      )}
    </AnimatePresence>
  );
}
