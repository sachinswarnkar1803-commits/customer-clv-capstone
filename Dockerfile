# Multi-stage production Dockerfile for BDS-34 Capstone Platform
FROM python:3.11-slim as base

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project source code and configurations
COPY configs/ ./configs/
COPY src/ ./src/
COPY dashboard/ ./dashboard/
COPY tests/ ./tests/
COPY docs/ ./docs/
COPY pytest.ini .
COPY README.md .

# Create necessary runtime directories
RUN mkdir -p data/raw data/processed data/synthetic data/validation models reports

# Expose Streamlit dashboard port
EXPOSE 8501

# Healthcheck
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Default command launches interactive Streamlit dashboard
ENTRYPOINT ["python", "-m", "streamlit", "run", "dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
