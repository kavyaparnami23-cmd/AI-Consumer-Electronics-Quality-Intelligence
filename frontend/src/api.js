import axios from 'axios';

const API_BASE = 'http://localhost:8000';

const client = axios.create({
  baseURL: API_BASE,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const getHealth = async () => {
  const res = await client.get('/health');
  return res.data;
};

export const getModels = async () => {
  const res = await client.get('/models');
  return res.data;
};

export const getSystemStatus = async () => {
  const res = await client.get('/system/status');
  return res.data;
};

export const predictClassic = async (features) => {
  const res = await client.post('/classic/predict', { features });
  return res.data;
};

export const predictClassicBatch = async (samples) => {
  const res = await client.post('/classic/predict/batch', { samples });
  return res.data;
};

export const predictDL = async (features, model = 'lstm') => {
  const res = await client.post('/dl/predict', { features, model });
  return res.data;
};

export const predictNLP = async (text, model = 'tfidf') => {
  const res = await client.post('/nlp/sentiment', { text, model });
  return res.data;
};

export const detectTSAnomaly = async (window) => {
  const res = await client.post('/timeseries/anomaly', { window });
  return res.data;
};

export const detectTSIsoForest = async (window) => {
  const res = await client.post('/timeseries/anomaly/isolation-forest', { window });
  return res.data;
};
