FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Install system dependencies if any are needed (e.g. build-essential, libffi-dev, etc.)
# We do this before pip install to build wheels if needed, but since we use slim, we might need gcc/libffi-dev for cryptography/coincurve.
# Let's install build dependencies.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all repository files into the image
COPY . .

CMD ["python", "live_trading_testnet.py"]
