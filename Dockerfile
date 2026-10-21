FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and UI assets
COPY . .

# Create persistent data directory
RUN mkdir -p /app/data

ENV HOST=0.0.0.0
ENV PORT=8000
ENV MEMORY_STORAGE_PATH=/app/data/livemind_memory.db

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "livemind.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
