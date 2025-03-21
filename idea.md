# Optimized Serverless MLOps Architecture with UV and Multi-Stage Builds

I'll enhance the previous design by using UV (a faster Python package installer) and multi-stage Docker builds to create a more efficient serverless architecture for the Iris dataset.

## Project Structure

```
optimized-serverless-mlops/
├── docker-compose.yml
├── function-runner/
│   ├── Dockerfile
│   ├── app.py
│   └── requirements.txt
├── functions/
│   ├── train/
│   │   ├── Dockerfile
│   │   ├── train.py
│   │   └── requirements.txt
│   └── predict/
│       ├── Dockerfile
│       ├── predict.py
│       └── requirements.txt
├── models/
│   └── .gitkeep
└── web-ui/
    └── index.html
```

## 1. Docker Compose Setup

```yaml
version: '3'
services:
  function-runner:
    build: ./function-runner
    ports:
      - "5000:5000"
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock
      - ./functions:/app/functions
      - ./models:/app/models
    environment:
      - PYTHONUNBUFFERED=1
    restart: always

  web-ui:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./web-ui:/usr/share/nginx/html
    depends_on:
      - function-runner
    restart: always
```

## 2. Function Runner with UV

Create `function-runner/Dockerfile`:

```dockerfile
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

# Install Docker
RUN apt-get update && \
    apt-get install -y docker.io --no-install-recommends && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /venv /venv

# Set environment to use the virtual environment
ENV PATH="/venv/bin:$PATH"

WORKDIR /app

COPY app.py .

# Create models directory
RUN mkdir -p /app/models

EXPOSE 5000
CMD ["python", "app.py"]
```

The Function Runner's `app.py` and `requirements.txt` remain the same as in the previous implementation.

## 3. Multi-Stage Training Function

Create `functions/train/Dockerfile`:

```dockerfile
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

COPY train.py .

CMD ["python", "train.py"]
```

Create `functions/train/requirements.txt`:

```
scikit-learn
pandas
numpy
joblib
```

The `train.py` file remains the same as in the previous implementation.

## 4. Multi-Stage Prediction Function

Create `functions/predict/Dockerfile`:

```dockerfile
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
```

Create `functions/predict/requirements.txt`:

```
scikit-learn
pandas
numpy
joblib
```

The `predict.py` file remains the same as in the previous implementation.

## 5. Enhanced Function Runner App

Let's update the `function-runner/app.py` to include better caching and efficiency:

```python
from flask import Flask, request, jsonify
import docker
import os
import json
import time
import uuid
import threading

app = Flask(__name__)

# Initialize Docker client
docker_client = docker.from_env()

# Directory where models are stored
MODELS_DIR = os.path.abspath('./models')

# Track image build status
image_build_locks = {}
build_lock = threading.Lock()

def ensure_image_built(image_name, function_dir):
    """Build the image if it doesn't exist, with proper locking to prevent parallel builds"""
    global image_build_locks
    
    # Check if image exists
    try:
        docker_client.images.get(image_name)
        return True  # Image exists
    except docker.errors.ImageNotFound:
        pass  # Need to build
    
    # Acquire a lock for this specific image
    with build_lock:
        if image_name in image_build_locks:
            build_lock_obj = image_build_locks[image_name]
        else:
            build_lock_obj = threading.Lock()
            image_build_locks[image_name] = build_lock_obj
    
    # Build with lock to prevent parallel builds of same image
    with build_lock_obj:
        # Check again in case another thread built while we were waiting
        try:
            docker_client.images.get(image_name)
            return True
        except docker.errors.ImageNotFound:
            pass
        
        print(f"Building image {image_name}...")
        build_start = time.time()
        
        try:
            _, build_logs = docker_client.images.build(
                path=function_dir,
                tag=image_name,
                rm=True,
                pull=True,  # Pull latest base images
                nocache=False  # Use cache for faster builds
            )
            
            for log in build_logs:
                if 'stream' in log and log['stream'].strip():
                    print(log['stream'].strip())
                    
            print(f"Image {image_name} built in {time.time() - build_start:.2f} seconds")
            return True
            
        except Exception as e:
            print(f"Error building image {image_name}: {e}")
            return False

@app.route('/invoke', methods=['POST'])
def invoke():
    data = request.get_json()
    
    # Extract function name and parameters
    function_name = data.get('function')
    params = data.get('params', {})
    
    if function_name not in ['train', 'predict']:
        return jsonify({
            'status': 'error',
            'error': f"Unknown function: {function_name}"
        }), 400
    
    # Generate a unique container name
    container_name = f"{function_name}-{uuid.uuid4().hex[:8]}"
    
    try:
        # Convert params to environment variables
        environment = {}
        if function_name == 'train':
            for key, value in params.items():
                if value is not None:  # Skip None values
                    environment[key.upper()] = str(value)
        elif function_name == 'predict':
            environment['FEATURES'] = json.dumps(params)
        
        # Path to function directory
        function_dir = os.path.abspath(f'./functions/{function_name}')
        
        # Ensure image is built
        image_name = f"serverless-mlops/{function_name}"
        if not ensure_image_built(image_name, function_dir):
            return jsonify({
                'status': 'error',
                'error': f"Failed to build image {image_name}"
            }), 500
        
        print(f"Starting container {container_name} for {function_name}")
        start_time = time.time()
        
        # Run the container
        container = docker_client.containers.run(
            image=image_name,
            name=container_name,
            environment=environment,
            volumes={MODELS_DIR: {'bind': '/app/models', 'mode': 'rw'}},
            detach=True
        )
        
        # Wait for the container to finish
        result = container.wait()
        exit_code = result['StatusCode']
        
        # Get logs from the container
        logs = container.logs().decode('utf-8')
        
        # Try to parse the output as JSON
        try:
            output = json.loads(logs)
        except json.JSONDecodeError:
            output = {"logs": logs}
        
        # Remove the container
        container.remove()
        
        # Prepare the response
        if exit_code != 0:
            return jsonify({
                'status': 'error',
                'error': logs,
                'exit_code': exit_code
            }), 500
        
        return jsonify({
            'status': 'success',
            'data': output,
            'execution_time': time.time() - start_time
        })
        
    except Exception as e:
        # Make sure to clean up the container if it exists
        try:
            container = docker_client.containers.get(container_name)
            container.remove(force=True)
        except:
            pass
        
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

## Benefits of These Optimizations

1. **UV Package Installation**:
   - Significantly faster dependency installation
   - Better dependency resolution
   - More efficient handling of Python packages

2. **Multi-Stage Docker Builds**:
   - Smaller final images (no build tools included)
   - Better layer caching
   - Improved security due to reduced attack surface

3. **Optimized Build Process**:
   - Proper locking to prevent parallel image builds of the same function
   - Improved caching strategy
   - Better status tracking and logging

4. **Container Efficiency**:
   - Virtual environments for clean dependency management
   - Minimized container size
   - Faster startup times

## Running the System

1. Create all directories and files as described
2. Start the services:

```bash
docker-compose up -d
```

3. Access the web UI at http://localhost

This optimized architecture maintains the true serverless behavior (creating and destroying containers on demand) while significantly improving build and execution performance through modern tools and techniques.