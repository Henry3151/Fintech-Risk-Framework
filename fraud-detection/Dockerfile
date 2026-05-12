# ── Stage 1: builder ────────────────────────────────────────────────────────
FROM python:3.14-slim AS builder

WORKDIR /app

# System deps for XGBoost / imbalanced-learn
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Stage 2: runtime ─────────────────────────────────────────────────────────
FROM python:3.14-slim AS runtime

WORKDIR /app

# Runtime-only system libraries (libgomp for XGBoost)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Source code
COPY src/ src/
COPY models/ models/

# Non-root user
RUN useradd -m appuser && chown -R appuser /app
USER appuser

# PYTHONPATH so `from domain.` imports work without editable install
ENV PYTHONPATH=/app/src
ENV MODELS_DIR=/app/models

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
