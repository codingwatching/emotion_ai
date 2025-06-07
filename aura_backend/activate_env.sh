#!/bin/bash
# Aura Backend Environment Activation Helper

if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
    echo "✅ Virtual environment activated"
    echo "Python: $(which python)"
    echo "Version: $(python --version)"
else
    echo "❌ Virtual environment not found. Run setup_uv.sh first."
fi
