#!/bin/bash

# Set the environment name (ensure it matches your environment.yml)
ENV_NAME="pd-ws24"

echo "Checking if Conda is installed..."
if ! command -v conda &> /dev/null
then
    echo "Error: Conda is not installed. Please install Anaconda or Miniconda first."
    exit 1
fi

echo "Creating or updating the Conda environment: $ENV_NAME..."
if conda env list | grep -q "$ENV_NAME"; then
    echo "Environment already exists. Updating..."
    conda env update --name "$ENV_NAME" --file environment.yml --prune
else
    echo "Environment not found. Creating a new one..."
    conda env create -f environment.yml
fi

echo "Activating the Conda environment..."
eval "$(conda shell.bash hook)"  # Ensure the script can activate conda
conda activate "$ENV_NAME"

# Check if requirements.txt exists
if [ -f "requirements.txt" ]; then
    echo "Installing additional pip dependencies..."
    pip install -r requirements.txt
else
    echo "No requirements.txt file found. Skipping pip installation."
fi

echo "Installation complete. To activate the environment, run:"
echo "conda activate $ENV_NAME"