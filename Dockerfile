# Use python 3.10 slim as base
FROM python:3.10-slim AS builder

WORKDIR /build

# System dependencies for compiling (like psycopg2) and ML models
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy dataset and training script
COPY Dataset/ ./Dataset/
COPY train.py .

# Train the model (Since .pkl files are massive and gitignored, we build them inside Docker)
# This generates ensemble.pkl and vectorizer.pkl
RUN python train.py

# --- Stage 2: Serve API ---
FROM python:3.10-slim

WORKDIR /app

# System dependencies for psycopg2
RUN apt-get update && apt-get install -y \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy API files and static frontend
COPY app.py database.py evaluate.py ./
COPY static/ ./static/

# Copy the trained models from the builder stage
COPY --from=builder /build/ensemble.pkl /build/vectorizer.pkl ./

# Set environment variable flag (optional usage)
ENV DOCKERIZED=true

# Expose port
EXPOSE 8000

# Start server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
