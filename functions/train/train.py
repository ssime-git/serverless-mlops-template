# functions/train/train.py
import os
import json
import sys
import time
import numpy as np
import mlflow
import logging
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_model():
    logger.info("Starting model training...")
    start_time = time.time()
    
    # Configure MLflow
    mlflow_uri = os.environ.get('MLFLOW_TRACKING_URI', 'http://mlflow:5000')
    logger.info(f"Setting MLflow tracking URI to: {mlflow_uri}")
    mlflow.set_tracking_uri(mlflow_uri)
    experiment_name = "iris_classification"
    
    # Get or create the experiment
    try:
        experiment_id = mlflow.create_experiment(experiment_name)
        logger.info(f"Created new experiment: {experiment_name} with ID: {experiment_id}")
    except:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        experiment_id = experiment.experiment_id
        logger.info(f"Using existing experiment: {experiment_name} with ID: {experiment_id}")
    
    mlflow.set_experiment(experiment_name)
    
    # Load environment variables for parameters
    n_estimators = int(os.environ.get('N_ESTIMATORS', '100'))
    max_depth_str = os.environ.get('MAX_DEPTH', 'None')
    max_depth = None if max_depth_str == 'None' else int(max_depth_str)
    test_size = float(os.environ.get('TEST_SIZE', '0.2'))
    random_state = int(os.environ.get('RANDOM_STATE', '42'))
    
    logger.info(f"Parameters: n_estimators={n_estimators}, max_depth={max_depth}, test_size={test_size}")
    
    # Load Iris dataset
    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names
    
    logger.info("Loaded Iris dataset")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    logger.info("Split data into training and testing sets")
    
    # Start an MLflow run
    with mlflow.start_run() as run:
        # Log parameters
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("random_state", random_state)
        
        logger.info("Logged parameters to MLflow")
        
        # Train model
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state
        )
        model.fit(X_train, y_train)
        
        logger.info("Trained model")
        
        # Evaluate model
        train_accuracy = model.score(X_train, y_train)
        test_accuracy = model.score(X_test, y_test)
        
        logger.info("Evaluated model")
        
        # Log metrics
        mlflow.log_metric("train_accuracy", train_accuracy)
        mlflow.log_metric("test_accuracy", test_accuracy)
        mlflow.log_metric("training_time", time.time() - start_time)
        
        logger.info("Logged metrics to MLflow")
        
        # Log model
        mlflow.sklearn.log_model(
            model, 
            "model",
            registered_model_name="iris_model",
            signature=mlflow.models.infer_signature(X_train, y_train),
            input_example=X_train[:5]
        )
        
        logger.info("Logged model to MLflow")
        
        # Log feature names and target names as artifacts
        with open("feature_names.json", "w") as f:
            # Convert to list if it's a numpy array, otherwise use as is
            feature_names_list = feature_names.tolist() if hasattr(feature_names, 'tolist') else list(feature_names)
            json.dump({"feature_names": feature_names_list}, f)
        
        with open("target_names.json", "w") as f:
            # Convert to list if it's a numpy array, otherwise use as is
            target_names_list = target_names.tolist() if hasattr(target_names, 'tolist') else list(target_names)
            json.dump({"target_names": target_names_list}, f)
            
        mlflow.log_artifact("feature_names.json")
        mlflow.log_artifact("target_names.json")
        
        logger.info("Logged feature names and target names as artifacts")
        
        # Return results as JSON
        result = {
            "run_id": run.info.run_id,
            "experiment_id": run.info.experiment_id,
            "model_uri": f"runs:/{run.info.run_id}/model",
            "metrics": {
                "train_accuracy": train_accuracy,
                "test_accuracy": test_accuracy,
                "training_time": time.time() - start_time
            },
            "parameters": {
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "test_size": test_size
            }
        }
        
        logger.info("Returning results as JSON")
        print(json.dumps(result))
        return 0

if __name__ == '__main__':
    try:
        exit_code = train_model()
        sys.exit(exit_code)
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        print(json.dumps({"error": str(e)}))
        sys.exit(1)