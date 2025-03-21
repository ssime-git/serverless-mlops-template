# Build stage
FROM python:3.9-slim AS builder

# Install UV for faster dependency installation
RUN pip install uv

WORKDIR /build

COPY requirements.txt .

# Use UV to install dependencies into a virtual environment
RUN uv venv /venv
RUN uv pip install --no-cache-dir -r requirements.txt --python /venv/bin/python

# Runtime stage
FROM python:3.9-slim

# Install Docker CLI
RUN apt-get update && apt-get install -y docker.io && apt-get clean

# Copy virtual environment from builder
COPY --from=builder /venv /venv

# Set environment to use the virtual environment
ENV PATH="/venv/bin:$PATH"

WORKDIR /app

COPY app.py .

# Create models directory
RUN mkdir -p /app/models
RUN mkdir -p /mlflow/artifacts
RUN chmod -R 777 /mlflow

EXPOSE 8000
CMD ["python", "app.py"]