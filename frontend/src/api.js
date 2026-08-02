import axios from 'axios';

// In development:  VITE_API_BASE is not set → uses '' which hits the vite proxy at /api/*
// In Docker/prod:  VITE_API_BASE is not set → uses '' which hits the nginx proxy at /api/*
// Override via:    VITE_API_BASE=http://localhost:8000 (if running frontend standalone)
const API_BASE = import.meta.env.VITE_API_BASE ?? '';

const client = axios.create({
  baseURL: API_BASE,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 30000,
});

// All paths are relative — nginx (or vite proxy) handles routing to backend
export const getHealth = async () => {
  const res = await client.get('/api/health');
  return res.data;
};

export const getModels = async () => {
  const res = await client.get('/api/models');
  return res.data;
};

export const getSystemStatus = async () => {
  const res = await client.get('/api/system/status');
  return res.data;
};

export const predictClassic = async (features) => {
  const res = await client.post('/api/classic/predict', { features });
  return res.data;
};

export const predictClassicBatch = async (samples) => {
  const res = await client.post('/api/classic/predict/batch', { samples });
  return res.data;
};

export const predictDL = async (features, model = 'lstm') => {
  const res = await client.post('/api/dl/predict', { features, model });
  return res.data;
};

export const predictNLP = async (text, model = 'tfidf') => {
  const res = await client.post('/api/nlp/sentiment', { text, model });
  return res.data;
};

export const detectTSAnomaly = async (window) => {
  const res = await client.post('/api/timeseries/anomaly', { window });
  return res.data;
};

export const detectTSIsoForest = async (window) => {
  const res = await client.post('/api/timeseries/anomaly/isolation-forest', { window });
  return res.data;
};
