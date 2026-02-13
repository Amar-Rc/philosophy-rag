#!/bin/bash

# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Install common dependencies
pip install -r requirements.txt

# Install dependencies for both backends
echo "Installing MLX and Ollama dependencies..."
pip install -r ../mlx_rag/requirements.txt
pip install -r ../ollama_rag/requirements.txt

echo "Setup complete. To run the app, activate the virtual environment with:"
echo "source venv/bin/activate"
echo "Then run the app with:"
echo "streamlit run app.py"
