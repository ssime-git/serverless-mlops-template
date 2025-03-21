#!/bin/bash

# Create logs directory if it doesn't exist
mkdir -p logs

# List the most recent log files
echo "=== Most recent log files ==="
ls -lt logs/ | head -n 10

# Check if a log type is provided
if [ $# -eq 0 ]; then
    echo "Usage: ./check_recent_logs.sh <log-type>"
    echo "Example: ./check_recent_logs.sh train"
    exit 1
fi

# Find the most recent log file for the specified type
LOG_TYPE=$1
RECENT_LOG=$(ls -t logs/${LOG_TYPE}* 2>/dev/null | head -n 1)

if [ -z "$RECENT_LOG" ]; then
    echo "No log files found for type: $LOG_TYPE"
    exit 1
fi

echo "=== Contents of most recent $LOG_TYPE log: $RECENT_LOG ==="
cat "$RECENT_LOG"
