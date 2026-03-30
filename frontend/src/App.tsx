import { useState } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import HomePage from '@/components/HomePage';
import Dashboard from '@/components/Dashboard';

type View = 'home' | 'dashboard';

export default function App() {
  const [view, setView] = useState<View>('home');

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
          <HomePage onStartInvestigation={() => setView('dashboard')} />
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
          <Dashboard onBack={() => setView('home')} />
        </motion.div>
      )}
    </AnimatePresence>
  );
}
