#!/bin/bash

# 1. Create the virtual environment
python3 -m venv rag_env

# 2. Use the python/pip paths directly from the env 
# This ensures installation happens inside the env without needing 'source'
./rag_env/bin/pip install --upgrade pip
./rag_env/bin/pip install -r requirements.txt

echo "🎉 Environment setup complete. To start working, run: source rag_env/bin/activate"