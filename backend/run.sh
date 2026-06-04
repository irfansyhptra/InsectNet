#!/bin/bash
# Start the Backend Server using the Virtual Environment

# Get the directory where the script is located (backend/)
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# Change to the root directory of the project
cd "$DIR/.."

echo "Starting Backend using Virtual Environment Python..."
.venv/bin/python backend/main.py
