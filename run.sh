#!/bin/bash

# Script to run all intelligence computation experiments

echo ">>> Starting IC Activity 3: Neural Networks and Neuro-Fuzzy Experiments"

# 1. Install requirements if needed
# pip install -r requirements.txt

# 2. Run analyses
echo "--- Running Diabetes Analysis ---"
python3 src/diabetesAnalise.py

echo "--- Running Credit-G Analysis ---"
python3 src/creditGAnalise.py

echo "--- Running Auto MPG Analysis ---"
python3 src/autoMpgAnalise.py

echo "--- Running Ames Housing Analysis ---"
python3 src/amesHousingAnalise.py

echo "--- Running Final Global Comparison ---"
python3 src/finalComparison.py

echo ">>> All experiments completed. Results can be found in the output/ directory."
