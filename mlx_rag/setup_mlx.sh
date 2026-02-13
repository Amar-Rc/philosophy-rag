#!/bin/bash

# 1. Create the native environment
python3 -m venv mlx_env

# 2. Install dependencies using the specific environment path
./mlx_env/bin/pip install --upgrade pip
./mlx_env/bin/pip install -r requirements_mlx.txt


echo "-------------------------------------------------------"
echo "✅ MLX Environment Ready!"
echo "To activate, run: source mlx_env/bin/activate"
echo "-------------------------------------------------------"