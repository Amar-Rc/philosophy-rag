#!/bin/bash

# This script downloads the data, embeddings, and models required to run the application.

# --- IMPORTANT ---
# 1. Create a single zip file containing the 'data', 'embeddings', and 'models' directories.
# 2. Upload the zip file to a hosting service (e.g., Google Drive, Dropbox).
# 3. Get a direct download link for the zip file.
# 4. Replace the placeholder link below with your direct download link.

DOWNLOAD_LINK="https://drive.google.com/uc?export=download&id=1TM4Pn0MXd-3aer5bzquTZy--qoz5eGH7"

# --- DO NOT EDIT BELOW THIS LINE ---

# Download the data
echo "Downloading data..."
curl -L -o data.zip "$DOWNLOAD_LINK"

# Unzip the data
echo "Unzipping data..."
unzip data.zip

# Clean up
echo "Cleaning up..."
rm data.zip

echo "Data download and setup complete."
