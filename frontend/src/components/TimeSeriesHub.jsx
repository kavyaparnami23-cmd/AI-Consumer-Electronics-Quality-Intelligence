import React, { useState } from 'react';
import { Activity, AlertTriangle, ShieldCheck, CheckCircle2, Sliders } from 'lucide-react';
import { detectTSAnomaly, detectTSIsoForest } from '../api';
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts';

export default function TimeSeriesHub() {
  const [modelType, setModelType] = useState('autoencoder'); // 'autoencoder' | 'isoforest'
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  // Generate 30 timesteps window (default normal vs anomalous preset)
  const generateNormalWindow = () => {
    return Array.from({ length: 30 }, (_, i) => [
      298.1 + Math.sin(i / 3) * 1.5,
      308.6 + Math.cos(i / 3) * 1.5,
      1550.0 + Math.random() * 20,
      42.0 + Math.random() * 2,
      i * 2.0,
      0.0,
      0.0,
      0.0,
    ]);
  };

  const generateAnomalyWindow = () => {
    return Array.from({ length: 30 }, (_, i) => [
      305.0 + i * 0.5,
      318.0 + i * 0.6,
      2850.0 + Math.random() * 100,
      68.0 + Math.random() * 10,
      210.0 + i * 2.0,
      1.0,
      0.0,
      0.0,
    ]);
  };

  const [windowData, setWindowData] = useState(generateNormalWindow());

  const handleRunDetection = async () => {
    setLoading(true);
    setError(null);
    try {
      const fn = modelType === 'autoencoder' ? detectTSAnomaly : detectTSIsoForest;
      const data = await fn(windowData);
      setResult(data);
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Anomaly detection failed');
    } finally {
      setLoading(false);
    }
  };

  // Graph data formatted for Recharts
  const chartData = windowData.map((row, idx) => ({
    timestep: t => `T+${idx + 1}`,
    airTemp: row[0],
    procTemp: row[1],
    speed: row[2] / 100, // Scale for visual overlay
    torque: row[3],
  }));

  return (
    <div className="space-y-6">
      {/* Header Banner */}
      <div className="glass-card rounded-xl p-6 border border-slate-800 flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h2 className="text-lg font-bold text-white flex items-center gap-2">
            <Activity className="w-5 h-5 text-amber-400" />
            Time-Series Anomaly Detection Stream
          </h2>
          <p className="text-slate-400 text-xs mt-1">
            Evaluates 30 sequential sensor timesteps using PyTorch LSTM Autoencoder reconstruction loss or Isolation Forest.
          </p>
        </div>

        {/* Controls */}
        <div className="flex items-center gap-3">
          <button
            onClick={() => setWindowData(generateNormalWindow())}
            className="px-3 py-1.5 rounded-lg bg-slate-900 border border-slate-800 text-xs font-semibold text-slate-300 hover:text-white"
          >
            Load Normal Preset
          </button>
          <button
            onClick={() => setWindowData(generateAnomalyWindow())}
            className="px-3 py-1.5 rounded-lg bg-rose-500/10 border border-rose-500/30 text-xs font-semibold text-rose-400 hover:bg-rose-500/20"
          >
            Load Anomaly Preset
          </button>
        </div>
      </div>

      {error && (
        <div className="flex items-center gap-3 p-4 rounded-xl bg-rose-500/10 border border-rose-500/30 text-rose-400 text-sm">
          <AlertTriangle className="w-5 h-5 flex-shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {/* 30-Timestep Signal Overlay Chart */}
      <div className="glass-card rounded-xl p-6 border border-slate-800">
        <h3 className="text-sm font-bold text-white mb-4 flex items-center gap-2">
          <Sliders className="w-4 h-4 text-indigo-400" />
          30-Timestep Sensor Telemetry Stream (Air Temp, Process Temp, Torque, Speed)
        </h3>

        <div className="h-64 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={chartData} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
              <XAxis dataKey="timestep" stroke="#64748b" fontSize={10} />
              <YAxis stroke="#64748b" fontSize={10} />
              <Tooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '0.75rem', color: '#fff' }} />
              <Line type="monotone" dataKey="airTemp" stroke="#38bdf8" name="Air Temp [K]" strokeWidth={2} dot={false} />
              <Line type="monotone" dataKey="procTemp" stroke="#818cf8" name="Process Temp [K]" strokeWidth={2} dot={false} />
              <Line type="monotone" dataKey="torque" stroke="#f59e0b" name="Torque [Nm]" strokeWidth={2} dot={false} />
              <Line type="monotone" dataKey="speed" stroke="#10b981" name="Speed (x100 rpm)" strokeWidth={2} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div className="mt-4 flex items-center justify-between">
          <div className="flex bg-slate-900 p-1.5 rounded-xl border border-slate-800 text-xs">
            <button
              onClick={() => setModelType('autoencoder')}
              className={`px-3 py-1.5 rounded-lg font-semibold transition-all ${
                modelType === 'autoencoder' ? 'bg-indigo-600 text-white' : 'text-slate-400 hover:text-white'
              }`}
            >
              LSTM Autoencoder
            </button>
            <button
              onClick={() => setModelType('isoforest')}
              className={`px-3 py-1.5 rounded-lg font-semibold transition-all ${
                modelType === 'isoforest' ? 'bg-indigo-600 text-white' : 'text-slate-400 hover:text-white'
              }`}
            >
              Isolation Forest
            </button>
          </div>

          <button
            onClick={handleRunDetection}
            disabled={loading}
            className="px-6 py-2.5 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white font-semibold text-sm transition-all shadow-lg shadow-indigo-600/30 disabled:opacity-50"
          >
            {loading ? 'Evaluating Reconstruction Loss...' : `Detect Anomaly via ${modelType === 'autoencoder' ? 'Autoencoder' : 'IsoForest'}`}
          </button>
        </div>
      </div>

      {/* Detection Results */}
      {result && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="glass-card rounded-xl p-6 border border-slate-800 flex items-center justify-between">
            <div>
              <span className="text-slate-400 text-xs font-semibold uppercase tracking-wider">Status</span>
              <h3 className={`text-2xl font-black mt-1 ${result.is_anomaly ? 'text-rose-400' : 'text-emerald-400'}`}>
                {result.is_anomaly ? 'ANOMALY DETECTED' : 'NORMAL PATTERN'}
              </h3>
            </div>
            <div className={`p-3 rounded-xl ${result.is_anomaly ? 'bg-rose-500/10 text-rose-400' : 'bg-emerald-500/10 text-emerald-400'}`}>
              {result.is_anomaly ? <AlertTriangle className="w-6 h-6" /> : <CheckCircle2 className="w-6 h-6" />}
            </div>
          </div>

          <div className="glass-card rounded-xl p-6 border border-slate-800">
            <span className="text-slate-400 text-xs font-semibold uppercase tracking-wider">Reconstruction Error / Score</span>
            <div className="text-2xl font-extrabold text-white mt-1">{result.anomaly_score?.toFixed(4)}</div>
            <p className="text-slate-500 text-xs mt-1">Raw MSE loss across sequence</p>
          </div>

          <div className="glass-card rounded-xl p-6 border border-slate-800">
            <span className="text-slate-400 text-xs font-semibold uppercase tracking-wider">Threshold Cutoff</span>
            <div className="text-2xl font-extrabold text-amber-400 mt-1">{result.threshold?.toFixed(4)}</div>
            <p className="text-slate-500 text-xs mt-1">Calibrated decision boundary</p>
          </div>
        </div>
      )}
    </div>
  );
}
