#!/bin/bash
# Start the Smart Insect Identifier Backend Server
# This script ensures that the virtual environment is activated before running the server

# Get the directory where the script is located
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

echo "Activating virtual environment..."
source .venv/bin/activate

echo "Starting Backend API..."
cd backend
python main.py
