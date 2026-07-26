import React, { useState, useEffect } from 'react';
import { getModels, getSystemStatus } from './api';
import DashboardOverview from './components/DashboardOverview';
import ClassicMlHub from './components/ClassicMlHub';
import DeepLearningHub from './components/DeepLearningHub';
import TimeSeriesHub from './components/TimeSeriesHub';
import NlpSentimentHub from './components/NlpSentimentHub';
import { LayoutDashboard, ShieldCheck, Cpu, Activity, MessageSquare, Terminal } from 'lucide-react';

export default function App() {
  const [activeTab, setActiveTab] = useState('overview'); // 'overview' | 'classic' | 'dl' | 'ts' | 'nlp'
  const [modelsStatus, setModelsStatus] = useState(null);
  const [systemStatus, setSystemStatus] = useState(null);
  const [loadingModels, setLoadingModels] = useState(false);
  const [predictionHistory, setPredictionHistory] = useState([]);

  const handlePredictionMade = (record) => {
    setPredictionHistory((prev) => [record, ...prev]);
  };

  const fetchStatus = async () => {
    setLoadingModels(true);
    try {
      const [modelsData, sysData] = await Promise.all([getModels(), getSystemStatus()]);
      setModelsStatus(modelsData);
      setSystemStatus(sysData);
    } catch (err) {
      console.error('Failed to fetch status:', err);
    } finally {
      setLoadingModels(false);
    }
  };

  useEffect(() => {
    fetchStatus();
  }, []);

  const navItems = [
    { id: 'overview', label: 'Overview & Analytics', icon: LayoutDashboard },
    { id: 'classic', label: 'Classic ML & SHAP', icon: ShieldCheck },
    { id: 'dl', label: 'Deep Learning', icon: Cpu },
    { id: 'ts', label: 'Time Series', icon: Activity },
    { id: 'nlp', label: 'NLP Sentiment', icon: MessageSquare },
  ];

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans">
      {/* Top Navbar */}
      <header className="border-b border-slate-800 bg-slate-900/80 backdrop-blur-md sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="p-2.5 rounded-xl bg-gradient-to-tr from-indigo-600 to-purple-600 shadow-lg shadow-indigo-600/30">
              <Cpu className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-lg font-bold text-white tracking-tight">AI Electronics Quality Intelligence</h1>
              <p className="text-xs text-slate-400">FastAPI ML Backend • React + Tailwind Dashboard</p>
            </div>
          </div>

          <a
            href="http://localhost:8000/docs"
            target="_blank"
            rel="noreferrer"
            className="flex items-center gap-2 px-3.5 py-1.5 rounded-xl bg-slate-800 hover:bg-slate-700 text-xs font-semibold text-slate-200 border border-slate-700 transition-all"
          >
            <Terminal className="w-4 h-4 text-indigo-400" />
            FastAPI Swagger UI
          </a>
        </div>
      </header>

      {/* Main Container */}
      <div className="max-w-7xl mx-auto px-6 py-8 flex-1 w-full space-y-8">
        {/* Navigation Tabs */}
        <nav className="flex items-center gap-2 overflow-x-auto pb-1 border-b border-slate-800/80">
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = activeTab === item.id;
            return (
              <button
                key={item.id}
                onClick={() => setActiveTab(item.id)}
                className={`flex items-center gap-2.5 px-4 py-2.5 rounded-xl text-sm font-semibold transition-all whitespace-nowrap ${
                  isActive
                    ? 'bg-indigo-600 text-white shadow-lg shadow-indigo-600/30'
                    : 'text-slate-400 hover:text-slate-200 bg-slate-900/40 hover:bg-slate-900 border border-transparent hover:border-slate-800'
                }`}
              >
                <Icon className={`w-4 h-4 ${isActive ? 'text-white' : 'text-slate-400'}`} />
                {item.label}
              </button>
            );
          })}
        </nav>

        {/* Tab Content */}
        <main className="transition-all duration-300">
          {activeTab === 'overview' && (
            <DashboardOverview
              modelsStatus={modelsStatus}
              systemStatus={systemStatus}
              loading={loadingModels}
              onRefresh={fetchStatus}
              history={predictionHistory}
            />
          )}
          {activeTab === 'classic' && <ClassicMlHub onPredictionMade={handlePredictionMade} />}
          {activeTab === 'dl' && <DeepLearningHub onPredictionMade={handlePredictionMade} />}
          {activeTab === 'ts' && <TimeSeriesHub onPredictionMade={handlePredictionMade} />}
          {activeTab === 'nlp' && <NlpSentimentHub onPredictionMade={handlePredictionMade} />}
        </main>
      </div>

      {/* Footer */}
      <footer className="border-t border-slate-900 bg-slate-950 py-6 text-center text-xs text-slate-600">
        AI Consumer Electronics Quality Intelligence System • Powered by PyTorch, Scikit-learn, XGBoost, DistilBERT & FastAPI
      </footer>
    </div>
  );
}
