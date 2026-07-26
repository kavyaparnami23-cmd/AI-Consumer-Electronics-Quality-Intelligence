import React, { useState } from 'react';
import { MessageSquare, Sparkles, AlertCircle, ThumbsUp, ThumbsDown, MinusCircle } from 'lucide-react';
import { predictNLP } from '../api';

export default function NlpSentimentHub() {
  const [model, setModel] = useState('distilbert'); // 'tfidf' | 'distilbert'
  const [text, setText] = useState(
    'The electronic build quality is outstanding! Heat dissipation is well controlled and the motor runs smoothly.'
  );
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const presets = [
    'The electronic build quality is outstanding! Heat dissipation is well controlled and the motor runs smoothly.',
    'Tool wear rate is dangerously high and the housing over-heated within 5 minutes of operation.',
    'Average quality product. Works as specified, nothing special.',
  ];

  const handleAnalyze = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const data = await predictNLP(text, model);
      setResult(data);
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Sentiment analysis failed');
    } finally {
      setLoading(false);
    }
  };

  const getSentimentIcon = (sentiment) => {
    const s = String(sentiment).toLowerCase();
    if (s.includes('pos') || s.includes('label_2')) return <ThumbsUp className="w-6 h-6 text-emerald-400" />;
    if (s.includes('neg') || s.includes('label_0')) return <ThumbsDown className="w-6 h-6 text-rose-400" />;
    return <MinusCircle className="w-6 h-6 text-amber-400" />;
  };

  return (
    <div className="space-y-6">
      {/* Header Banner */}
      <div className="glass-card rounded-xl p-6 border border-slate-800 flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h2 className="text-lg font-bold text-white flex items-center gap-2">
            <MessageSquare className="w-5 h-5 text-cyan-400" />
            Customer Feedback Sentiment Analytics
          </h2>
          <p className="text-slate-400 text-xs mt-1">
            Analyze customer reviews and feedback comments using TF-IDF + Logistic Regression or fine-tuned DistilBERT transformer model.
          </p>
        </div>

        {/* Model Switcher */}
        <div className="flex bg-slate-900 p-1.5 rounded-xl border border-slate-800 text-xs font-semibold">
          <button
            onClick={() => setModel('tfidf')}
            className={`px-4 py-2 rounded-lg transition-all ${
              model === 'tfidf' ? 'bg-cyan-600 text-white shadow-md' : 'text-slate-400 hover:text-white'
            }`}
          >
            TF-IDF + Logistic Regression
          </button>
          <button
            onClick={() => setModel('distilbert')}
            className={`px-4 py-2 rounded-lg transition-all ${
              model === 'distilbert' ? 'bg-cyan-600 text-white shadow-md' : 'text-slate-400 hover:text-white'
            }`}
          >
            DistilBERT Transformer
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
        {/* Input Text Form */}
        <form onSubmit={handleAnalyze} className="lg:col-span-7 glass-card rounded-xl p-6 border border-slate-800 space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-bold text-white flex items-center gap-2">
              <Sparkles className="w-4 h-4 text-cyan-400" />
              Customer Feedback Text
            </h3>
            <span className="text-xs text-slate-500">Max 128 tokens</span>
          </div>

          <textarea
            rows={5}
            value={text}
            onChange={(e) => setText(e.target.value)}
            className="w-full p-4 rounded-xl bg-slate-900 border border-slate-800 text-sm text-slate-200 focus:outline-none focus:border-cyan-500"
            placeholder="Type customer review text here..."
          />

          <div className="flex items-center gap-2 overflow-x-auto pb-1">
            <span className="text-xs text-slate-500 font-medium">Sample Presets:</span>
            {presets.map((preset, idx) => (
              <button
                key={idx}
                type="button"
                onClick={() => setText(preset)}
                className="px-2.5 py-1 rounded-lg bg-slate-900 border border-slate-800 text-xs text-slate-400 hover:text-white truncate max-w-[180px]"
              >
                Sample #{idx + 1}
              </button>
            ))}
          </div>

          <button
            type="submit"
            disabled={loading}
            className="w-full py-2.5 rounded-xl bg-cyan-600 hover:bg-cyan-500 text-white font-semibold text-sm transition-all shadow-lg shadow-cyan-600/30 disabled:opacity-50"
          >
            {loading ? 'Classifying Sentiment via Model...' : `Analyze Sentiment using ${model.toUpperCase()}`}
          </button>
        </form>

        {/* Sentiment Result Card */}
        <div className="lg:col-span-5">
          {result ? (
            <div className="glass-card rounded-xl p-6 border border-slate-800 space-y-6">
              <div className="flex items-center justify-between">
                <div>
                  <span className="text-slate-400 text-xs font-semibold uppercase tracking-wider">Model Used</span>
                  <h4 className="text-lg font-bold text-cyan-400 mt-0.5">{result.model_used} Classifier</h4>
                </div>
                <div className="p-3 rounded-xl bg-slate-900 border border-slate-800">
                  {getSentimentIcon(result.sentiment)}
                </div>
              </div>

              <div>
                <span className="text-slate-400 text-xs font-semibold uppercase tracking-wider">Predicted Sentiment</span>
                <h3 className="text-2xl font-black text-white mt-1 capitalize">{result.sentiment}</h3>
              </div>

              <div className="space-y-2">
                <div className="flex justify-between text-xs font-medium">
                  <span className="text-slate-400">Classification Confidence</span>
                  <span className="text-white font-bold">{(result.confidence * 100).toFixed(1)}%</span>
                </div>
                <div className="w-full bg-slate-900 h-3 rounded-full overflow-hidden border border-slate-800">
                  <div
                    className="bg-cyan-500 h-full rounded-full transition-all duration-500"
                    style={{ width: `${result.confidence * 100}%` }}
                  ></div>
                </div>
              </div>
            </div>
          ) : (
            <div className="glass-card rounded-xl p-12 text-center text-slate-500 border border-slate-800">
              Submit review text to classify customer sentiment.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
