# ─────────────────────────────────────────────
# Stage 1 — Base image
# ─────────────────────────────────────────────
FROM python:3.10-slim AS base

# Prevents Python from writing .pyc files
ENV PYTHONDONTWRITEBYTECODE=1
# Ensures stdout/stderr are flushed immediately
ENV PYTHONUNBUFFERED=1

# System dependencies for TensorFlow + slim image
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ─────────────────────────────────────────────
# Stage 2 — Dependencies
# ─────────────────────────────────────────────
FROM base AS dependencies

WORKDIR /app

# Copy only requirements first (layer caching)
COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ─────────────────────────────────────────────
# Stage 3 — Final image
# ─────────────────────────────────────────────
FROM dependencies AS final

WORKDIR /app

# Copy application source
COPY app.py .

# Copy model and preprocessors
COPY trained_model/        ./trained_model/
COPY preprocessing_models/ ./preprocessing_models/

# Create a non-root user for security
RUN addgroup --system appgroup && \
    adduser  --system --ingroup appgroup appuser && \
    chown -R appuser:appgroup /app

USER appuser

# Expose Streamlit default port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Run the app
ENTRYPOINT ["streamlit", "run", "app.py", \
            "--server.port=8501", \
            "--server.address=0.0.0.0", \
            "--server.headless=true", \
            "--browser.gatherUsageStats=false"]