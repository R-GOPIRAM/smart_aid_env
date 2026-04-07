# ─── Base Image ────────────────────────────────────────────────────────────────
FROM python:3.10-slim

# ─── Metadata ─────────────────────────────────────────────────────────────────
LABEL maintainer="SmartAid Team" \
      version="1.0.0" \
      description="SmartAid-Env — Crisis-Aware OpenEnv Logistics Simulator"

# ─── System Dependencies ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ─── Working Directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ─── Install Python Dependencies ──────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ─── Copy Application ─────────────────────────────────────────────────────────
COPY . .

# ─── Environment ──────────────────────────────────────────────────────────────
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=7860

# ─── Expose Port ──────────────────────────────────────────────────────────────
EXPOSE 7860

# ─── Health Check ─────────────────────────────────────────────────────────────
HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# ─── Launch ───────────────────────────────────────────────────────────────────
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1", "--log-level", "info"]
