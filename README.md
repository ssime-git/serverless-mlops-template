# Serverless MLOps Template

A simple serverless MLOps architecture using Docker containers for on-demand execution of ML training and prediction functions, with MLflow for experiment tracking and model versioning.

## Architecture

This project implements a serverless MLOps architecture with the following components:

- **Function Runner**: A Flask API that dynamically creates Docker containers for ML functions
- **MLflow**: For experiment tracking and model registry
- **Web UI**: A simple dashboard to interact with the system
- **NGINX**: As an API gateway and static file server

## Features

- Train ML models with hyperparameter tuning
- Track experiments and model versions with MLflow
- Make predictions via API
- Web UI for easy interaction
- Containerized execution for true serverless behavior

## Components

### Function Runner

The function runner is a Flask application that:
- Receives requests to execute ML functions
- Dynamically builds Docker images if needed
- Creates containers to run the functions
- Returns the results
- Provides endpoints to interact with MLflow

### ML Functions

- **Train**: Trains a Random Forest model on the Iris dataset
- **Predict**: Makes predictions using a trained model

### MLflow Integration

MLflow is used for:
- Experiment tracking
- Model versioning
- Model registry
- Artifact storage

## Getting Started

### Prerequisites

- Docker and Docker Compose
- Git

### Running the Project

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/serverless-mlops-template.git
   cd serverless-mlops-template
   ```

2. Start the services:
   ```
   docker-compose up -d
   ```

3. Access the Web UI:
   - Open http://localhost in your browser

4. Access MLflow UI:
   - Open http://localhost/mlflow in your browser

## Usage

### Training a Model

Use the Web UI to train a model with custom hyperparameters:
1. Set the number of estimators, max depth, and test size
2. Click "Train Model"
3. View the results, including metrics and model URI

### Making Predictions

Use the Web UI to make predictions:
1. Enter the feature values
2. Click "Make Prediction"
3. View the prediction result

### Debugging and Monitoring

The system includes several features to help with debugging and monitoring:

1. **Container Logs**: All container outputs are saved to the `logs` directory with the naming convention `logs/{container_name}.log`

2. **Log Viewing Tools**:
   - `./view_logs.sh <container-name>`: View logs for a specific container
   - `./check_recent_logs.sh <log-type>`: View the most recent logs for a specific type (e.g., train, predict)

3. **Enhanced Error Handling**: The function runner provides detailed error information when containers fail, including the container name and log file location

4. **MLflow Dashboard**: Access the MLflow UI at http://localhost/mlflow to view experiment tracking, model versions, and metrics

## Architecture Details

- **Docker Containers**: Each function execution creates a new container
- **MLflow**: Tracks experiments, parameters, metrics, and models
- **Multistage Docker Builds**: Uses uv for faster dependency installation
- **NGINX**: Routes requests and serves static files

## License

This project is licensed under the MIT License - see the LICENSE file for details.