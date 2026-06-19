# ==========================================
# Stage 1: Build python dependencies
# ==========================================
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies to compile python packages if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install packages into /install directory
RUN mkdir /install && \
    pip install --no-cache-dir --prefix=/install -r requirements.txt

# ==========================================
# Stage 2: Final lightweight runner image
# ==========================================
FROM python:3.11-slim

WORKDIR /app

# Copy installed dependencies from the builder stage
COPY --from=builder /install /usr/local

# Prevent Python from writing .pyc files to disk and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Explicitly copy only the required running scripts to prevent bloating
COPY live_trading_testnet.py .
COPY entrypoint.py .

# Run entrypoint.py to launch both ETH and BTC bots
CMD ["python", "entrypoint.py"]
