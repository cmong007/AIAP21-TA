#!/usr/bin/env bash
# Exit immediately on error
set -e

#create data folder
mkdir -p data

# download db file if missing
if [ ! -f data/gas_monitoring.db ]; then
  curl -fSL \
    https://techassessment.blob.core.windows.net/aiap21-assessment-data/gas_monitoring.db \
    -o data/gas_monitoring.db
fi

# 1) Execute EDA notebook to generate cleaned_data.csv
jupyter nbconvert --to notebook --execute eda.ipynb --inplace

# Run the ML pipeline script
python src/mlp.py


