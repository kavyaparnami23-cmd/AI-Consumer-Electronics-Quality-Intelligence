# ── Backend Dockerfile ────────────────────────────────────────────────────────
FROM python:3.10-slim

# Minimal system deps (no GPU libraries needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Install Python deps (split for better layer caching) ──────────────────────

# Step 1: Install CPU-only torch FIRST (avoids pulling 3GB of CUDA packages)
RUN pip install --no-cache-dir \
    torch==2.3.1 \
    --index-url https://download.pytorch.org/whl/cpu

# Step 2: Install transformers (now torch is already present, no GPU deps pulled)
RUN pip install --no-cache-dir transformers>=4.30.0

# Step 3: Install the rest of the requirements
COPY requirements.txt .
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt

# ── Copy source code ──────────────────────────────────────────────────────────
COPY . .

# Create runtime directories
RUN mkdir -p saved_models mlruns logs

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]