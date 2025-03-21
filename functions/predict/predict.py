# functions/predict/predict.py
import os
import json
import sys
import time
import numpy as np
import mlflow
import logging
from mlflow.tracking import MlflowClient

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def predict():
    start_time = time.time()
    logger.info("Starting prediction...")
    
    try:
        # Get model name and version from environment variables
        model_name = os.environ.get('MODEL_NAME', 'iris_model')
        model_version = os.environ.get('MODEL_VERSION', 'latest')
        
        # Configure MLflow
        mlflow_uri = os.environ.get('MLFLOW_TRACKING_URI', 'http://mlflow:5000')
        logger.info(f"Setting MLflow tracking URI to: {mlflow_uri}")
        mlflow.set_tracking_uri(mlflow_uri)
        
        # Get features from environment variable
        features_json = os.environ.get('FEATURES', '[]')
        try:
            features = json.loads(features_json)
            logger.info(f"Features: {features}")
        except json.JSONDecodeError:
            error_msg = f"Invalid JSON in FEATURES environment variable: {features_json}"
            logger.error(error_msg)
            return {"error": error_msg}, 1
        
        logger.info(f"Predicting with model: {model_name} (version: {model_version})")
        
        # Load the model from MLflow
        model = load_model(model_name, model_version)
        if isinstance(model, tuple) and len(model) == 2 and isinstance(model[0], dict) and "error" in model[0]:
            # This is an error response
            return model
        
        # Prepare input features
        input_features = prepare_features(features)
        if isinstance(input_features, tuple) and len(input_features) == 2 and isinstance(input_features[0], dict) and "error" in input_features[0]:
            # This is an error response
            return input_features
        
        # Make prediction
        logger.info("Making prediction...")
        try:
            prediction = model.predict(input_features)
            logger.info("Prediction made successfully")
        except Exception as e:
            logger.error(f"Failed to make prediction: {e}")
            return {"error": f"Failed to make prediction: {e}"}, 1
        
        # Get class names
        class_names = get_class_names(model_name, model_version)
        
        # Format results
        result = format_prediction_result(prediction, class_names, start_time)
        
        logger.info(f"Prediction result: {result}")
        print(json.dumps(result))
        return result, 0
        
    except Exception as e:
        error_msg = f"Error during prediction: {str(e)}"
        logger.error(error_msg)
        print(json.dumps({"error": error_msg}))
        return {"error": error_msg}, 1

def load_model(model_name, model_version):
    """Load the model from MLflow"""
    try:
        # Set model URI
        if model_version == 'latest':
            model_uri = f"models:/{model_name}/latest"
        else:
            model_uri = f"models:/{model_name}/{model_version}"
            
        logger.info(f"Loading model from: {model_uri}")
        
        # Try to load the model using the MLflow model registry
        try:
            model = mlflow.pyfunc.load_model(model_uri)
            logger.info("Model loaded successfully from registry")
            return model
        except Exception as e:
            logger.error(f"Failed to load model from registry: {e}")
            
            # If that fails, try to find the model in the local MLflow artifacts directory
            try:
                # Get the MLflow client
                client = MlflowClient()
                
                # Find the latest version of the model
                if model_version == 'latest':
                    model_versions = client.get_latest_versions(model_name)
                    if not model_versions:
                        return {"error": f"No versions found for model {model_name}"}, 1
                    
                    # Get the run ID from the latest version
                    run_id = model_versions[0].run_id
                    logger.info(f"Found run_id: {run_id} for latest version of model {model_name}")
                else:
                    # Get the specific version
                    model_version_obj = client.get_model_version(model_name, model_version)
                    run_id = model_version_obj.run_id
                    logger.info(f"Found run_id: {run_id} for version {model_version} of model {model_name}")
                
                # Check what's in the artifacts directory
                logger.info(f"Checking MLflow artifacts directory...")
                try:
                    import subprocess
                    result = subprocess.run(['ls', '-la', '/mlflow/artifacts'], capture_output=True, text=True)
                    logger.info(f"MLflow artifacts directory contents: {result.stdout}")
                except Exception as ls_error:
                    logger.warning(f"Could not list MLflow artifacts directory: {ls_error}")
                
                # Try to load the model directly from the artifacts directory
                # Try multiple possible paths
                possible_paths = [
                    f"/mlflow/artifacts/{run_id}/artifacts/model",
                    f"/mlflow/artifacts/1/{run_id}/artifacts/model",
                    f"/mlflow/artifacts/{run_id}/model",
                    f"/mlflow/artifacts/1/{run_id}/model"
                ]
                
                for path in possible_paths:
                    logger.info(f"Trying to load model from path: {path}")
                    try:
                        # Check if the path exists
                        import os
                        if os.path.exists(path):
                            logger.info(f"Path {path} exists, attempting to load model")
                            model = mlflow.pyfunc.load_model(path)
                            logger.info(f"Model loaded successfully from {path}")
                            return model
                        else:
                            logger.warning(f"Path {path} does not exist")
                    except Exception as path_error:
                        logger.warning(f"Failed to load model from {path}: {path_error}")
                
                # If we get here, all paths failed
                raise Exception(f"Failed to load model from all attempted paths")
            except Exception as inner_e:
                logger.error(f"Failed to load model from local path: {inner_e}")
                return {"error": f"Failed to load model: {e}. Local path attempt failed: {inner_e}"}, 1
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return {"error": f"Error loading model: {e}"}, 1

def prepare_features(features):
    """Prepare input features for prediction"""
    try:
        logger.info(f"Preparing input features from: {type(features)}")
        if isinstance(features, dict):
            # If features provided as a dictionary with feature names
            # Convert to a 2D array for the model
            logger.info(f"Features as dict: {features}")
            feature_values = [features.get(f, 0.0) for f in ["sepal_length", "sepal_width", "petal_length", "petal_width"]]
            input_features = np.array([feature_values])
            logger.info(f"Converted to array: {input_features}")
            return input_features
        elif isinstance(features, list):
            # If features provided as a list, convert to numpy array
            logger.info(f"Features as list: {features}")
            if all(isinstance(item, (int, float)) for item in features):
                # Single instance as a 1D list, convert to 2D
                input_features = np.array([features])
                logger.info(f"Converted to 2D array: {input_features}")
                return input_features
            else:
                # Already in 2D format or other format, try to convert
                input_features = np.array(features)
                logger.info(f"Converted to array: {input_features}")
                if input_features.ndim == 1:
                    input_features = input_features.reshape(1, -1)
                    logger.info(f"Reshaped to 2D: {input_features}")
                return input_features
        else:
            error_msg = f"Unsupported feature format: {type(features)}"
            logger.error(error_msg)
            return {"error": error_msg}, 1
    except Exception as e:
        logger.error(f"Error preparing features: {e}")
        return {"error": f"Error preparing features: {e}"}, 1

def get_class_names(model_name, model_version):
    """Get class names from the model"""
    class_names = ["setosa", "versicolor", "virginica"]  # Default Iris class names
    
    try:
        # Get the MLflow client
        client = MlflowClient()
        
        # Find the model version
        if model_version == 'latest':
            model_versions = client.get_latest_versions(model_name)
            if not model_versions:
                logger.warning(f"No versions found for model {model_name}")
                return class_names
            
            run_id = model_versions[0].run_id
        else:
            model_version_obj = client.get_model_version(model_name, model_version)
            run_id = model_version_obj.run_id
        
        # Get target_names.json artifact
        try:
            artifact_path = client.download_artifacts(run_id, "target_names.json")
            with open(artifact_path, "r") as f:
                target_data = json.load(f)
                if "target_names" in target_data:
                    class_names = target_data["target_names"]
        except Exception as e:
            logger.warning(f"Could not load class names from model: {e}")
    except Exception as e:
        logger.warning(f"Error getting class names: {e}")
    
    return class_names

def format_prediction_result(prediction, class_names, start_time):
    """Format the prediction result"""
    prediction_list = prediction.tolist()
    
    if len(prediction_list) == 1:
        # Single prediction
        predicted_class = int(prediction_list[0])
        class_name = class_names[predicted_class] if predicted_class < len(class_names) else f"Class {predicted_class}"
        
        return {
            "prediction": predicted_class,
            "class_name": class_name,
            "prediction_time": time.time() - start_time
        }
    else:
        # Multiple predictions
        return {
            "predictions": prediction_list,
            "class_names": [class_names[int(p)] if int(p) < len(class_names) else f"Class {int(p)}" for p in prediction_list],
            "prediction_time": time.time() - start_time
        }

if __name__ == '__main__':
    try:
        result, exit_code = predict()
        
        sys.exit(exit_code)
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)