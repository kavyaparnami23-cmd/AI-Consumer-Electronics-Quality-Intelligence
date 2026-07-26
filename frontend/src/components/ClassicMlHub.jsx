import React, { useState } from 'react';
import { ShieldCheck, BarChart3, AlertCircle, Sparkles, Layers, CheckCircle, Upload, FileText } from 'lucide-react';
import { predictClassic, predictClassicBatch } from '../api';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';

export default function ClassicMlHub({ onPredictionMade }) {
  const [mode, setMode] = useState('single'); // 'single' | 'batch' | 'csv'
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  // Single sample form
  const [form, setForm] = useState({
    air_temperature: 298.1,
    process_temperature: 308.6,
    rotational_speed: 1551.0,
    torque: 42.8,
    tool_wear: 0.0,
    type_H: 0.0,
    type_L: 0.0,
    type_M: 1.0,
  });

  // Batch sample text
  const [batchJson, setBatchJson] = useState(
    JSON.stringify(
      [
        {
          air_temperature: 298.1,
          process_temperature: 308.6,
          rotational_speed: 1551.0,
          torque: 42.8,
          tool_wear: 0.0,
          type_H: 0.0,
          type_L: 0.0,
          type_M: 1.0,
        },
        {
          air_temperature: 302.5,
          process_temperature: 312.1,
          rotational_speed: 2860.0,
          torque: 68.4,
          tool_wear: 215.0,
          type_H: 1.0,
          type_L: 0.0,
          type_M: 0.0,
        },
      ],
      null,
      2
    )
  );
  const [batchResult, setBatchResult] = useState(null);
  const [csvFileName, setCsvFileName] = useState(null);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setForm((prev) => ({ ...prev, [name]: parseFloat(value) || 0 }));
  };

  const handlePredictSingle = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const data = await predictClassic(form);
      setResult(data);
      if (onPredictionMade) {
        onPredictionMade({
          timestamp: new Date().toLocaleTimeString(),
          model: 'Classic ML (XGB)',
          label: data.label,
          confidence: (data.confidence * 100).toFixed(1),
          isFailure: data.prediction === 1,
        });
      }
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Prediction failed');
    } finally {
      setLoading(false);
    }
  };

  const handlePredictBatch = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const samples = JSON.parse(batchJson);
      const data = await predictClassicBatch(samples);
      setBatchResult(data);
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Batch prediction failed');
    } finally {
      setLoading(false);
    }
  };

  // CSV File Upload handler
  const handleCsvFileUpload = (e) => {
    const file = e.target.files[0];
    if (!file) return;
    setCsvFileName(file.name);

    const reader = new FileReader();
    reader.onload = (event) => {
      const text = event.target.result;
      const lines = text.split('\n').filter((l) => l.trim());
      if (lines.length <= 1) return;

      const headers = lines[0].split(',').map((h) => h.trim().toLowerCase());
      const parsedSamples = [];

      for (let i = 1; i < lines.length; i++) {
        const values = lines[i].split(',').map((v) => parseFloat(v.trim()) || 0);
        if (values.length === headers.length) {
          const rowObj = {};
          headers.forEach((h, idx) => {
            rowObj[h] = values[idx];
          });
          parsedSamples.push(rowObj);
        }
      }

      setBatchJson(JSON.stringify(parsedSamples, null, 2));
    };
    reader.readAsText(file);
  };

  // Convert SHAP dict to Recharts array
  const shapData = result?.shap_values
    ? Object.entries(result.shap_values)
        .map(([feature, val]) => ({
          feature,
          val,
          absVal: Math.abs(val),
        }))
        .sort((a, b) => b.absVal - a.absVal)
    : [];

  return (
    <div className="space-y-6">
      {/* Tab Mode Switcher */}
      <div className="flex items-center gap-3 border-b border-slate-800 pb-4">
        <button
          onClick={() => setMode('single')}
          className={`flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-semibold transition-all ${
            mode === 'single'
              ? 'bg-indigo-600 text-white shadow-lg shadow-indigo-600/30'
              : 'text-slate-400 hover:text-white bg-slate-900/60'
          }`}
        >
          <Sparkles className="w-4 h-4" />
          Single Sensor Predictor + SHAP
        </button>
        <button
          onClick={() => setMode('batch')}
          className={`flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-semibold transition-all ${
            mode === 'batch'
              ? 'bg-indigo-600 text-white shadow-lg shadow-indigo-600/30'
              : 'text-slate-400 hover:text-white bg-slate-900/60'
          }`}
        >
          <Layers className="w-4 h-4" />
          Batch Prediction (JSON)
        </button>
        <button
          onClick={() => setMode('csv')}
          className={`flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-semibold transition-all ${
            mode === 'csv'
              ? 'bg-indigo-600 text-white shadow-lg shadow-indigo-600/30'
              : 'text-slate-400 hover:text-white bg-slate-900/60'
          }`}
        >
          <Upload className="w-4 h-4" />
          CSV File Upload
        </button>
      </div>

      {error && (
        <div className="flex items-center gap-3 p-4 rounded-xl bg-rose-500/10 border border-rose-500/30 text-rose-400 text-sm">
          <AlertCircle className="w-5 h-5 flex-shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {mode === 'single' ? (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          {/* Form Side */}
          <form onSubmit={handlePredictSingle} className="lg:col-span-5 glass-card rounded-xl p-6 border border-slate-800 space-y-4">
            <h2 className="text-base font-bold text-white flex items-center gap-2 mb-2">
              <ShieldCheck className="w-5 h-5 text-indigo-400" />
              Input Telemetry Features
            </h2>

            <div className="grid grid-cols-2 gap-3 text-xs">
              <div>
                <label className="block text-slate-400 font-medium mb-1">Air Temp [K]</label>
                <input
                  type="number"
                  step="0.1"
                  name="air_temperature"
                  value={form.air_temperature}
                  onChange={handleChange}
                  className="w-full px-3 py-2 rounded-lg bg-slate-900 border border-slate-800 text-white focus:outline-none focus:border-indigo-500"
                />
              </div>
              <div>
                <label className="block text-slate-400 font-medium mb-1">Process Temp [K]</label>
                <input
                  type="number"
                  step="0.1"
                  name="process_temperature"
                  value={form.process_temperature}
                  onChange={handleChange}
                  className="w-full px-3 py-2 rounded-lg bg-slate-900 border border-slate-800 text-white focus:outline-none focus:border-indigo-500"
                />
              </div>
              <div>
                <label className="block text-slate-400 font-medium mb-1">Rotational Speed [rpm]</label>
                <input
                  type="number"
                  step="1"
                  name="rotational_speed"
                  value={form.rotational_speed}
                  onChange={handleChange}
                  className="w-full px-3 py-2 rounded-lg bg-slate-900 border border-slate-800 text-white focus:outline-none focus:border-indigo-500"
                />
              </div>
              <div>
                <label className="block text-slate-400 font-medium mb-1">Torque [Nm]</label>
                <input
                  type="number"
                  step="0.1"
                  name="torque"
                  value={form.torque}
                  onChange={handleChange}
                  className="w-full px-3 py-2 rounded-lg bg-slate-900 border border-slate-800 text-white focus:outline-none focus:border-indigo-500"
                />
              </div>
              <div>
                <label className="block text-slate-400 font-medium mb-1">Tool Wear [min]</label>
                <input
                  type="number"
                  step="1"
                  name="tool_wear"
                  value={form.tool_wear}
                  onChange={handleChange}
                  className="w-full px-3 py-2 rounded-lg bg-slate-900 border border-slate-800 text-white focus:outline-none focus:border-indigo-500"
                />
              </div>
              <div>
                <label className="block text-slate-400 font-medium mb-1">Product Type</label>
                <select
                  value={form.type_H ? 'H' : form.type_L ? 'L' : 'M'}
                  onChange={(e) => {
                    const v = e.target.value;
                    setForm((prev) => ({
                      ...prev,
                      type_H: v === 'H' ? 1.0 : 0.0,
                      type_L: v === 'L' ? 1.0 : 0.0,
                      type_M: v === 'M' ? 1.0 : 0.0,
                    }));
                  }}
                  className="w-full px-3 py-2 rounded-lg bg-slate-900 border border-slate-800 text-white focus:outline-none focus:border-indigo-500"
                >
                  <option value="M">Medium (Type M)</option>
                  <option value="L">Low (Type L)</option>
                  <option value="H">High (Type H)</option>
                </select>
              </div>
            </div>

            <button
              type="submit"
              disabled={loading}
              className="w-full py-2.5 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white font-semibold text-sm transition-all shadow-lg shadow-indigo-600/30 disabled:opacity-50 mt-2"
            >
              {loading ? 'Running Inference & SHAP...' : 'Predict Sensor Failure & Explain'}
            </button>
          </form>

          {/* Results Side */}
          <div className="lg:col-span-7 space-y-6">
            {result ? (
              <>
                <div className="glass-card rounded-xl p-6 border border-slate-800 flex items-center justify-between">
                  <div>
                    <span className="text-slate-400 text-xs font-semibold uppercase tracking-wider font-mono">Classification</span>
                    <h3 className={`text-2xl font-black mt-1 ${result.prediction === 1 ? 'text-rose-400' : 'text-emerald-400'}`}>
                      {result.label}
                    </h3>
                  </div>
                  <div className="text-right">
                    <span className="text-slate-400 text-xs font-semibold uppercase tracking-wider font-mono">Confidence Score</span>
                    <div className="text-2xl font-extrabold text-white mt-1">{(result.confidence * 100).toFixed(1)}%</div>
                  </div>
                </div>

                {/* SHAP Explanation Bar Chart */}
                <div className="glass-card rounded-xl p-6 border border-slate-800 space-y-4">
                  <h3 className="text-sm font-bold text-white flex items-center gap-2">
                    <BarChart3 className="w-4 h-4 text-indigo-400" />
                    SHAP Feature Contribution Score Breakdown
                  </h3>

                  <div className="h-64 w-full">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={shapData} layout="vertical" margin={{ top: 5, right: 20, left: 110, bottom: 5 }}>
                        <XAxis type="number" stroke="#64748b" fontSize={11} />
                        <YAxis dataKey="feature" type="category" stroke="#94a3b8" fontSize={11} tick={{ fill: '#cbd5e1' }} />
                        <Tooltip
                          contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '0.75rem', color: '#fff' }}
                        />
                        <Bar dataKey="val" radius={[0, 4, 4, 0]}>
                          {shapData.map((entry, index) => (
                            <Cell key={`cell-${index}`} fill={entry.val >= 0 ? '#ef4444' : '#10b981'} />
                          ))}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </div>

                  {/* Plain Language SHAP Impact Text */}
                  {result.top_features && result.top_features.length > 0 && (
                    <div className="p-4 rounded-xl bg-slate-900/80 border border-slate-800 text-xs space-y-2">
                      <div className="font-bold text-white uppercase tracking-wider">Top Explainability Insights:</div>
                      {result.top_features.slice(0, 3).map((item, i) => {
                        const val = item.shap_value ?? item.impact ?? 0;
                        const formattedVal = typeof val === 'number' ? val.toFixed(4) : String(val);
                        return (
                          <div key={i} className="flex items-center gap-2 text-slate-300">
                            <span className={`w-2 h-2 rounded-full ${val >= 0 ? 'bg-rose-500' : 'bg-emerald-500'}`}></span>
                            <span className="font-mono text-white font-semibold">{item.feature}</span>:
                            <span>
                              {val >= 0
                                ? `Increased failure probability by +${formattedVal}`
                                : `Decreased failure risk by ${formattedVal}`}
                            </span>
                          </div>
                        );
                      })}
                    </div>
                  )}
                </div>
              </>
            ) : (
              <div className="glass-card rounded-xl p-12 text-center text-slate-500 border border-slate-800">
                Submit telemetry inputs on the left to view real-time predictions and SHAP explanations.
              </div>
            )}
          </div>
        </div>
      ) : mode === 'csv' ? (
        /* CSV Upload Mode */
        <div className="glass-card rounded-xl p-6 border border-slate-800 space-y-6">
          <div className="text-center p-8 border-2 border-dashed border-slate-800 rounded-xl hover:border-indigo-500 transition-all bg-slate-900/40">
            <Upload className="w-10 h-10 text-indigo-400 mx-auto mb-3" />
            <h3 className="text-sm font-bold text-white">Upload Telemetry CSV File</h3>
            <p className="text-slate-400 text-xs mt-1 mb-4">
              Select a CSV containing sensor reading columns (`air_temperature`, `process_temperature`, `rotational_speed`, `torque`, `tool_wear`).
            </p>
            <label className="px-5 py-2.5 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white font-semibold text-xs cursor-pointer inline-block transition-all shadow-lg shadow-indigo-600/30">
              Browse CSV File
              <input type="file" accept=".csv" onChange={handleCsvFileUpload} className="hidden" />
            </label>
            {csvFileName && (
              <div className="mt-3 text-xs text-emerald-400 font-medium flex items-center justify-center gap-1.5">
                <FileText className="w-4 h-4" />
                Loaded: {csvFileName}
              </div>
            )}
          </div>

          {batchJson && (
            <div className="space-y-4">
              <button
                onClick={handlePredictBatch}
                disabled={loading}
                className="px-6 py-2.5 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white font-semibold text-sm transition-all shadow-lg shadow-indigo-600/30 disabled:opacity-50"
              >
                {loading ? 'Processing CSV Telemetry Batch...' : 'Run Batch Inference on CSV Data'}
              </button>
            </div>
          )}

          {batchResult && (
            <div className="space-y-4 border-t border-slate-800 pt-4">
              <h3 className="text-sm font-bold text-white flex items-center gap-2">
                <CheckCircle className="w-4 h-4 text-emerald-400" />
                Batch CSV Prediction Results ({batchResult.count} samples)
              </h3>

              <div className="overflow-x-auto">
                <table className="w-full text-xs text-left text-slate-300">
                  <thead className="text-slate-400 uppercase bg-slate-900/80 border-b border-slate-800">
                    <tr>
                      <th className="px-4 py-3">Sample #</th>
                      <th className="px-4 py-3">Classification</th>
                      <th className="px-4 py-3">Confidence</th>
                    </tr>
                  </thead>
                  <tbody>
                    {batchResult.predictions.map((pred, i) => (
                      <tr key={i} className="border-b border-slate-800/50 hover:bg-slate-900/40">
                        <td className="px-4 py-3 font-semibold text-white">#{i + 1}</td>
                        <td className="px-4 py-3">
                          <span className={`px-2.5 py-1 rounded-full text-xs font-bold ${
                            pred === 1 ? 'bg-rose-500/10 text-rose-400 border border-rose-500/30' : 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/30'
                          }`}>
                            {batchResult.labels[i]}
                          </span>
                        </td>
                        <td className="px-4 py-3 text-slate-200">{(batchResult.confidences[i] * 100).toFixed(1)}%</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      ) : (
        /* JSON Batch Mode */
        <div className="glass-card rounded-xl p-6 border border-slate-800 space-y-6">
          <form onSubmit={handlePredictBatch} className="space-y-4">
            <h2 className="text-base font-bold text-white flex items-center gap-2">
              <Layers className="w-5 h-5 text-indigo-400" />
              Batch Input (JSON Array of SensorFeatures objects or vectors)
            </h2>

            <textarea
              rows={8}
              value={batchJson}
              onChange={(e) => setBatchJson(e.target.value)}
              className="w-full p-4 rounded-xl bg-slate-900 border border-slate-800 text-xs font-mono text-emerald-300 focus:outline-none focus:border-indigo-500"
            />

            <button
              type="submit"
              disabled={loading}
              className="px-6 py-2.5 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white font-semibold text-sm transition-all shadow-lg shadow-indigo-600/30 disabled:opacity-50"
            >
              {loading ? 'Processing Batch...' : 'Run Batch Prediction'}
            </button>
          </form>

          {batchResult && (
            <div className="space-y-4 border-t border-slate-800 pt-4">
              <h3 className="text-sm font-bold text-white flex items-center gap-2">
                <CheckCircle className="w-4 h-4 text-emerald-400" />
                Batch Results ({batchResult.count} samples)
              </h3>

              <div className="overflow-x-auto">
                <table className="w-full text-xs text-left text-slate-300">
                  <thead className="text-slate-400 uppercase bg-slate-900/80 border-b border-slate-800">
                    <tr>
                      <th className="px-4 py-3">Sample #</th>
                      <th className="px-4 py-3">Classification</th>
                      <th className="px-4 py-3">Confidence</th>
                    </tr>
                  </thead>
                  <tbody>
                    {batchResult.predictions.map((pred, i) => (
                      <tr key={i} className="border-b border-slate-800/50 hover:bg-slate-900/40">
                        <td className="px-4 py-3 font-semibold text-white">#{i + 1}</td>
                        <td className="px-4 py-3">
                          <span className={`px-2.5 py-1 rounded-full text-xs font-bold ${
                            pred === 1 ? 'bg-rose-500/10 text-rose-400 border border-rose-500/30' : 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/30'
                          }`}>
                            {batchResult.labels[i]}
                          </span>
                        </td>
                        <td className="px-4 py-3 text-slate-200">{(batchResult.confidences[i] * 100).toFixed(1)}%</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
