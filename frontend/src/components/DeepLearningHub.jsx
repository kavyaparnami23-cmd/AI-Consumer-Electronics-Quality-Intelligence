import React, { useState } from 'react';
import { Cpu, Zap, AlertCircle, CheckCircle2 } from 'lucide-react';
import { predictDL } from '../api';

export default function DeepLearningHub() {
  const [modelType, setModelType] = useState('lstm'); // 'lstm' | 'cnn'
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  // 8 sensor feature inputs
  const [features, setFeatures] = useState([0.1, -0.2, 0.5, 0.3, -0.1, 0.8, 0.4, 0.2]);

  const handleFeatureChange = (index, value) => {
    const next = [...features];
    next[index] = parseFloat(value) || 0;
    setFeatures(next);
  };

  const handlePredict = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const data = await predictDL(features, modelType);
      setResult(data);
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Deep learning inference failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="glass-card rounded-xl p-6 border border-slate-800 flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h2 className="text-lg font-bold text-white flex items-center gap-2">
            <Cpu className="w-5 h-5 text-indigo-400" />
            Deep Learning Sensor Fault Detector
          </h2>
          <p className="text-slate-400 text-xs mt-1">
            Choose between Bidirectional LSTM with Attention or 1-D Convolutional Neural Network (CNN).
          </p>
        </div>

        {/* Model Selector */}
        <div className="flex bg-slate-900 p-1.5 rounded-xl border border-slate-800">
          <button
            onClick={() => setModelType('lstm')}
            className={`px-4 py-2 rounded-lg text-xs font-semibold transition-all ${
              modelType === 'lstm' ? 'bg-indigo-600 text-white shadow-md' : 'text-slate-400 hover:text-white'
            }`}
          >
            LSTM (Sequence)
          </button>
          <button
            onClick={() => setModelType('cnn')}
            className={`px-4 py-2 rounded-lg text-xs font-semibold transition-all ${
              modelType === 'cnn' ? 'bg-indigo-600 text-white shadow-md' : 'text-slate-400 hover:text-white'
            }`}
          >
            1-D CNN (Conv)
          </button>
        </div>
      </div>

      {error && (
        <div className="flex items-center gap-3 p-4 rounded-xl bg-rose-500/10 border border-rose-500/30 text-rose-400 text-sm">
          <AlertCircle className="w-5 h-5 flex-shrink-0" />
          <span>{error}</span>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Form Inputs */}
        <form onSubmit={handlePredict} className="lg:col-span-6 glass-card rounded-xl p-6 border border-slate-800 space-y-4">
          <h3 className="text-sm font-bold text-white flex items-center gap-2 mb-2">
            <Zap className="w-4 h-4 text-indigo-400" />
            8-Channel Sensor Signal Vector
          </h3>

          <div className="grid grid-cols-2 gap-3 text-xs">
            {features.map((val, idx) => (
              <div key={idx}>
                <label className="block text-slate-400 font-medium mb-1">Sensor Channel #{idx + 1}</label>
                <input
                  type="number"
                  step="0.01"
                  value={val}
                  onChange={(e) => handleFeatureChange(idx, e.target.value)}
                  className="w-full px-3 py-2 rounded-lg bg-slate-900 border border-slate-800 text-white focus:outline-none focus:border-indigo-500 font-mono"
                />
              </div>
            ))}
          </div>

          <button
            type="submit"
            disabled={loading}
            className="w-full py-2.5 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white font-semibold text-sm transition-all shadow-lg shadow-indigo-600/30 disabled:opacity-50 mt-2"
          >
            {loading ? 'Executing Neural Network Forward Pass...' : `Run ${modelType.toUpperCase()} Inference`}
          </button>
        </form>

        {/* Prediction Results */}
        <div className="lg:col-span-6">
          {result ? (
            <div className="glass-card rounded-xl p-6 border border-slate-800 space-y-6">
              <div className="flex items-center justify-between">
                <div>
                  <span className="text-slate-400 text-xs font-semibold uppercase tracking-wider">Model Used</span>
                  <h4 className="text-xl font-bold text-indigo-400 mt-0.5">{result.model_used} Model</h4>
                </div>

                <span className={`px-3 py-1.5 rounded-full text-xs font-extrabold uppercase border ${
                  result.prediction === 1 ? 'bg-rose-500/10 text-rose-400 border-rose-500/30' : 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30'
                }`}>
                  {result.label}
                </span>
              </div>

              <div className="space-y-2">
                <div className="flex justify-between text-xs font-medium">
                  <span className="text-slate-400">Failure Probability</span>
                  <span className="text-white font-bold">{(result.confidence * 100).toFixed(1)}%</span>
                </div>
                <div className="w-full bg-slate-900 h-3 rounded-full overflow-hidden border border-slate-800">
                  <div
                    className={`h-full rounded-full transition-all duration-500 ${
                      result.prediction === 1 ? 'bg-rose-500' : 'bg-emerald-500'
                    }`}
                    style={{ width: `${result.confidence * 100}%` }}
                  ></div>
                </div>
              </div>

              <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800 text-xs text-slate-300 space-y-1">
                <div className="font-semibold text-white">Execution Telemetry</div>
                <div>Deep Neural Network Architecture: {result.model_used === 'LSTM' ? '2-Layer BiLSTM + Softmax' : '3-Layer 1D Conv + MaxPool'}</div>
                <div>Prediction Output Class: {result.prediction}</div>
              </div>
            </div>
          ) : (
            <div className="glass-card rounded-xl p-12 text-center text-slate-500 border border-slate-800">
              Run forward pass to evaluate deep learning sensor failure risk.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
