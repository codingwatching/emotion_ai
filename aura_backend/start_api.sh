#!/bin/bash

echo "🚀 Starting Aura API Server..."

# Activate virtual environment
source .venv/bin/activate

# Set Python path to include current directory
PYTHONPATH="${PYTHONPATH}:$(pwd)"
export PYTHONPATH

# Run the main application
python -m main
