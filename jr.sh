#!/bin/bash

# Create directories
mkdir -p \
data/raw/defective \
data/raw/non_defective \
data/processed/train \
data/processed/val \
data/processed/test \
notebooks \
src \
models \
reports

# Create files
touch notebooks/defect_classification.ipynb
touch src/main.py
touch models/defect_classifier.h5
touch reports/report.pdf
touch requirements.txt
touch README.md
touch .gitignore

echo "Project structure created successfully!"