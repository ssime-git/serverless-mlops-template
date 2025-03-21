#!/bin/bash

# Create logs directory if it doesn't exist
mkdir -p logs

# Check if a container name is provided
if [ $# -eq 0 ]; then
    echo "Usage: ./view_logs.sh <container-name>"
    echo "Available logs:"
    ls -la logs/
    exit 1
fi

# View the logs for the specified container
CONTAINER_NAME=$1
LOG_FILE="logs/${CONTAINER_NAME}.log"

if [ -f "$LOG_FILE" ]; then
    echo "=== Logs for container $CONTAINER_NAME ==="
    cat "$LOG_FILE"
else
    echo "Log file $LOG_FILE not found."
    echo "Available logs:"
    ls -la logs/
fi
