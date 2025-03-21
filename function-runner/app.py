from flask import Flask, request, jsonify
import docker
import os
import json
import time
import uuid
import threading
import mlflow
from mlflow.tracking import MlflowClient
import requests
import logging
import re

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)

# Set MLflow tracking URI from environment variable
mlflow_uri = os.environ.get('MLFLOW_TRACKING_URI', 'http://mlflow:5000')
logger.info(f"Setting MLflow tracking URI to: {mlflow_uri}")
mlflow.set_tracking_uri(mlflow_uri)

# Initialize Flask app
app = Flask(__name__)

# Initialize Docker client
docker_client = docker.from_env()

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
        
        logger.info(f"Building image {image_name}...")
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
                    logger.info(log['stream'].strip())
                    
            logger.info(f"Image {image_name} built in {time.time() - build_start:.2f} seconds")
            return True
            
        except Exception as e:
            logger.error(f"Error building image {image_name}: {e}")
            return False

@app.route('/invoke', methods=['POST'])
def invoke():
    data = request.get_json()
    
    if not data:
        return jsonify({
            'status': 'error',
            'error': 'Invalid JSON payload or Content-Type header missing'
        }), 400
    
    # Extract function name and parameters
    function_name = data.get('function')
    params = data.get('params', {})
    
    logger.info(f"Received invoke request for function: {function_name}, params: {params}")
    
    if function_name not in ['train', 'predict']:
        return jsonify({
            'status': 'error',
            'error': f"Unknown function: {function_name}"
        }), 400
    
    # Generate a unique container name
    container_name = f"{function_name}-{uuid.uuid4().hex[:8]}"
    
    try:
        # Convert params to environment variables
        environment = {
            'MLFLOW_TRACKING_URI': mlflow_uri  # Pass MLflow tracking URI to the container
        }
        
        if function_name == 'train':
            for key, value in params.items():
                if value is not None:  # Skip None values
                    environment[key.upper()] = str(value)
        elif function_name == 'predict':
            # For prediction, we need model name/version and features
            # Check if params is a list (direct features) or a dictionary
            if isinstance(params, list):
                # If params is a list, it's the features directly
                environment['FEATURES'] = json.dumps(params)
            else:
                # For dictionary params, extract model info and features
                environment['MODEL_NAME'] = params.get('model_name', 'iris_model')
                environment['MODEL_VERSION'] = str(params.get('model_version', 'latest'))
                
                # Convert features to JSON string
                features = params.get('features', {})
                if features:
                    environment['FEATURES'] = json.dumps(features)
        
        # Path to function directory
        function_dir = os.path.abspath(f'./functions/{function_name}')
        
        # Ensure image is built
        image_name = f"serverless-mlops/{function_name}"
        if not ensure_image_built(image_name, function_dir):
            return jsonify({
                'status': 'error',
                'error': f"Failed to build image {image_name}"
            }), 500
        
        logger.info(f"Starting container {container_name} for {function_name}")
        start_time = time.time()
        
        # Run the container
        container = docker_client.containers.run(
            image=image_name,
            name=container_name,
            environment=environment,
            network="serverless-mlops-template_default",  # Connect to the same network as MLflow
            volumes={
                os.path.abspath('./mlflow_data'): {'bind': '/mlflow', 'mode': 'ro'}  # Mount MLflow data as read-only
            },
            detach=True
        )
        
        # Wait for the container to finish
        result = container.wait()
        exit_code = result['StatusCode']
        
        # Get logs from the container
        logs = container.logs().decode('utf-8')
        logger.info(f"Container {container_name} finished with exit code {exit_code} in {time.time() - start_time:.2f} seconds")
        
        # Save logs to file
        with open(f'logs/{container_name}.log', 'w') as f:
            f.write(logs)
        
        # Remove the container
        container.remove()
        
        # Process the output
        if exit_code != 0:
            logger.error(f"Function execution failed with exit code {exit_code}")
            logger.error(f"Container logs: {logs}")
            
            # Save detailed error information
            error_info = {
                'status': 'error',
                'error': f"Function execution failed with exit code {exit_code}",
                'logs': logs,
                'container_name': container_name,
                'log_file': f'logs/{container_name}.log'
            }
            
            return jsonify(error_info), 500
        
        # Try to parse output as JSON
        try:
            # Check if the output looks like HTML (starts with <!DOCTYPE or <html)
            if logs.strip().startswith(('<html', '<!DOCTYPE')):
                logger.warning("Container output appears to be HTML instead of JSON")
                
                # Extract any JSON that might be embedded in the HTML
                # This is a simple approach - look for anything between { and }
                json_matches = re.findall(r'({.*?})', logs, re.DOTALL)
                
                if json_matches:
                    # Try each potential JSON match
                    for json_str in json_matches:
                        try:
                            output = json.loads(json_str)
                            logger.info(f"Successfully extracted JSON from HTML: {json_str}")
                            output['status'] = 'success'
                            return jsonify(output)
                        except:
                            continue
                
                # If we couldn't extract JSON, return a generic success
                return jsonify({
                    'status': 'success',
                    'message': 'Operation completed successfully',
                    'container_name': container_name,
                    'log_file': f'logs/{container_name}.log'
                })
            
            # Normal JSON parsing
            output = json.loads(logs)
            output['status'] = 'success'
            return jsonify(output)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse container output as JSON: {e}")
            logger.error(f"Raw output: {logs}")
            
            # Return error with raw logs
            return jsonify({
                'status': 'error',
                'error': f"Failed to parse container output as JSON: {e}",
                'raw_output': logs,
                'container_name': container_name,
                'log_file': f'logs/{container_name}.log'
            }), 500
        
    except Exception as e:
        logger.error(f"Error invoking function {function_name}: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@app.route('/models', methods=['GET'])
def list_models():
    """List all models in MLflow"""
    try:
        # Get all registered models
        client = MlflowClient()
        models = client.search_registered_models()
        
        return jsonify({
            'status': 'success',
            'models': [
                {
                    'name': model.name,
                    'latest_versions': [
                        {
                            'version': version.version,
                            'stage': version.current_stage,
                            'run_id': version.run_id,
                            'creation_timestamp': version.creation_timestamp
                        } for version in model.latest_versions
                    ]
                } for model in models
            ]
        })
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@app.route('/models/<model_name>/versions', methods=['GET'])
def list_model_versions(model_name):
    """List all versions of a specific model"""
    try:
        client = MlflowClient()
        versions = client.get_latest_versions(model_name)
        
        return jsonify({
            'status': 'success',
            'model_name': model_name,
            'versions': [
                {
                    'version': version.version,
                    'stage': version.current_stage,
                    'run_id': version.run_id,
                    'creation_timestamp': version.creation_timestamp,
                    'description': version.description
                } for version in versions
            ]
        })
    except Exception as e:
        logger.error(f"Error listing model versions for {model_name}: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@app.route('/models/<model_name>/versions/<version>', methods=['GET'])
def get_model_version(model_name, version):
    """Get details for a specific model version"""
    try:
        client = MlflowClient()
        model_version = client.get_model_version(model_name, version)
        
        # Get the run to fetch metrics and parameters
        run = mlflow.get_run(model_version.run_id)
        
        return jsonify({
            'status': 'success',
            'model_version': {
                'name': model_version.name,
                'version': model_version.version,
                'stage': model_version.current_stage,
                'run_id': model_version.run_id,
                'creation_timestamp': model_version.creation_timestamp,
                'last_updated_timestamp': model_version.last_updated_timestamp,
                'description': model_version.description,
                'metrics': run.data.metrics,
                'parameters': run.data.params
            }
        })
    except Exception as e:
        logger.error(f"Error getting model version {version} for {model_name}: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@app.route('/experiments', methods=['GET'])
def list_experiments():
    """List all MLflow experiments"""
    try:
        experiments = mlflow.search_experiments()
        
        return jsonify({
            'status': 'success',
            'experiments': [
                {
                    'experiment_id': exp.experiment_id,
                    'name': exp.name,
                    'artifact_location': exp.artifact_location,
                    'lifecycle_stage': exp.lifecycle_stage,
                    'creation_time': exp.creation_time
                } for exp in experiments
            ]
        })
    except Exception as e:
        logger.error(f"Error listing experiments: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@app.route('/experiments/<experiment_id>/runs', methods=['GET'])
def list_runs(experiment_id):
    """List runs for a specific experiment"""
    try:
        runs = mlflow.search_runs(experiment_ids=[experiment_id])
        
        # Convert DataFrame to dict for JSON serialization
        runs_list = []
        for _, run in runs.iterrows():
            run_dict = run.to_dict()
            # Clean up any non-serializable objects
            for key, value in run_dict.items():
                if not isinstance(value, (str, int, float, bool, list, dict, type(None))):
                    run_dict[key] = str(value)
            runs_list.append(run_dict)
        
        return jsonify({
            'status': 'success',
            'experiment_id': experiment_id,
            'runs': runs_list
        })
    except Exception as e:
        logger.error(f"Error listing runs for experiment {experiment_id}: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@app.route('/debug', methods=['GET'])
def debug():
    """Debug endpoint to check system status"""
    try:
        # Check Docker
        docker_info = {
            'images': [img.tags for img in docker_client.images.list() if img.tags],
            'containers': [
                {
                    'name': container.name,
                    'image': container.image.tags[0] if container.image.tags else 'unknown',
                    'status': container.status
                } 
                for container in docker_client.containers.list(all=True)
            ]
        }
        
        # Check MLflow
        try:
            mlflow_status = requests.get(f"{mlflow_uri}/api/2.0/mlflow/experiments/list").json()
            mlflow_healthy = True
        except Exception as e:
            mlflow_status = str(e)
            mlflow_healthy = False
        
        return jsonify({
            'status': 'success',
            'timestamp': time.time(),
            'docker': docker_info,
            'mlflow': {
                'uri': mlflow_uri,
                'healthy': mlflow_healthy,
                'status': mlflow_status
            },
            'environment': {k: v for k, v in os.environ.items() if not k.startswith('_')}
        })
    except Exception as e:
        logger.error(f"Error in debug endpoint: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': time.time()
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=False)