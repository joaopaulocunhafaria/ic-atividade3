#!/bin/bash

# Script to run all experiments for the IC project

echo "Starting IC Activity 3 Experiments..."

# 1. Diabetes (Classification)
echo "Running Diabetes analysis..."
python3 src/diabetesAnalise.py

# 2. Credit-G (Classification)
echo "Running Credit-G analysis..."
python3 src/creditGAnalise.py

# 3. Auto MPG (Regression)
echo "Running Auto MPG analysis..."
python3 src/autoMpgAnalise.py

# 4. Ames Housing (Regression)
echo "Running Ames Housing analysis..."
python3 src/amesHousingAnalise.py

echo "All experiments completed. Results are in the output/ directory."
