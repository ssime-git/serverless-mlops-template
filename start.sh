#!/bin/bash

# Create logs directory if it doesn't exist
echo "Creating logs directory..."
mkdir -p logs

# Start fresh by stopping and removing all containers and volumes
echo "Stopping and removing all containers and volumes..."
docker-compose down -v
docker system prune -af --volumes

# Build and start the services
echo "Building and starting services..."
docker-compose up -d

echo "Waiting for services to initialize (20 seconds)..."
sleep 20

echo "Services are up and running!"
echo "Web UI: http://localhost"
echo "MLflow UI: http://localhost/mlflow"
echo ""
echo "To train a model, use the Web UI or make a POST request to http://localhost/invoke"
echo "To make predictions, use the Web UI or make a POST request to http://localhost/invoke"
echo ""
echo "To view logs, use ./view_logs.sh <container-name> or ./check_recent_logs.sh <log-type>"
