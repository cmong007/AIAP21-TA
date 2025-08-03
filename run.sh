#!/usr/bin/env bash
# Exit immediately on error
set -e

# Activate the virtual environment (Windows Git Bash)
source venv/Scripts/activate

# 1) Execute EDA notebook to generate cleaned_data.csv
jupyter nbconvert --to notebook --execute eda.ipynb --inplace

# Run the ML pipeline script
python src/mlp.py


