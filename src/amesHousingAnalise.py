import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neural_network import MLPRegressor

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.ames_housing import get_ames_housing_data
from src.experimentEngine import ExperimentEngine
from src.utils.fuzzyModels import TSKRegressor

def performEda(df, outputDir):
    print("\n--- Exploratory Data Analysis: Ames Housing ---")
    print(f"Samples: {df.shape[0]}, Features: {df.shape[1]-1}")
    
    # Target distribution
    plt.figure(figsize=(8, 6))
    sns.histplot(df['SalePrice'], kde=True)
    plt.title("SalePrice Distribution")
    plt.savefig(f"{outputDir}/edaTargetDist.png")
    plt.close()

def main():
    datasetName = "ames_housing"
    outputDir = f"output/{datasetName}"
    os.makedirs(outputDir, exist_ok=True)
    
    # 1. Load Data
    df = get_ames_housing_data()
    
    # 2. EDA
    performEda(df, outputDir)
    
    # 3. Prepare Features and Target
    X = df.drop(columns=['SalePrice'])
    y = df['SalePrice']
    
    # 4. Define Models
    models = {
        'RNA_MLP_Simple': {
            'class': MLPRegressor,
            'params': {'hidden_layer_sizes': (100,), 'activation': 'relu', 'max_iter': 5000, 'random_state': 42}
        },
        'RNA_MLP_Deep': {
            'class': MLPRegressor,
            'params': {'hidden_layer_sizes': (50, 25, 10), 'activation': 'relu', 'max_iter': 5000, 'random_state': 42}
        },
        'NeuroFuzzy_TSK_5': {
            'class': TSKRegressor,
            'params': {'nClusters': 5, 'm': 2.0}
        },
        'NeuroFuzzy_TSK_10': {
            'class': TSKRegressor,
            'params': {'nClusters': 10, 'm': 2.0}
        }
    }
    
    # 5. Run Experiments
    engine = ExperimentEngine(datasetName, taskType='regression')
    engine.runExperiments(X, y, models, numRepetitions=21)

if __name__ == "__main__":
    main()
