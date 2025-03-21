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

# Copy virtual environment from builder
COPY --from=builder /venv /venv

# Set environment to use the virtual environment
ENV PATH="/venv/bin:$PATH"

WORKDIR /app

COPY predict.py .

CMD ["python", "predict.py"]