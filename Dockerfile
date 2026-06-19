# Use an official lightweight Python runtime as a parent image
FROM python:3.11-slim


WORKDIR /app

# Prevent Python from writing .pyc files to disk and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
# Install system dependencies if any are needed (e.g. build-essential, libffi-dev, etc.)
# We do this before pip install to build wheels if needed, but since we use slim, we might need gcc/libffi-dev for cryptography/coincurve.
# Let's install build dependencies.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application script to the container
COPY live_trading_testnet.py .
# Copy all repository files into the image
COPY . .

# Run entrypoint.py to launch both ETH and BTC bots
CMD ["python", "entrypoint.py"]
