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
from src.utils.rbfModels import RBFRegressor

def performEda(df, outputDir):
    """
    Performs Exploratory Data Analysis for the Ames Housing dataset.
    """
    print("\n--- Exploratory Data Analysis: Ames Housing ---")
    
    # Target distribution
    plt.figure(figsize=(8, 6))
    sns.histplot(df['SalePrice'], kde=True)
    plt.title("SalePrice Distribution - Ames Housing")
    plt.savefig(f"{outputDir}/edaTargetDist.png")
    plt.close()
    
    # descriptive stats
    df.describe().to_csv(f"{outputDir}/descriptiveStats.csv")

def main():
    datasetName = "ames_housing"
    outputDir = f"output/{datasetName}"
    os.makedirs(outputDir, exist_ok=True)
    
    # 1. Load Data
    df = get_ames_housing_data()
    performEda(df, outputDir)
    
    # 2. Prepare Features and Target
    X = df.drop(columns=['SalePrice'])
    y = df['SalePrice']
    
    # 3. Define Model Configurations and Parameter Grids
    modelConfigs = {
        'MLP': {
            'class': MLPRegressor,
            'paramGrid': {
                'hidden_layer_sizes': [(100,), (50, 50)],
                'activation': ['relu'],
                'max_iter': [5000],
                'random_state': [42]
            }
        },
        'RBF': {
            'class': RBFRegressor,
            'paramGrid': {
                'nCenters': [10, 50, 100],
                'gamma': [0.01, 0.1, 1.0]
            }
        },
        'TSK_Variation_1': {
            'class': TSKRegressor,
            'paramGrid': {
                'nClusters': [5, 10],
                'm': [2.0]
            }
        },
        'TSK_Variation_2': {
            'class': TSKRegressor,
            'paramGrid': {
                'nClusters': [15, 20],
                'm': [1.5, 2.0]
            }
        }
    }
    
    # 4. Run Systematic Experiments
    engine = ExperimentEngine(datasetName, taskType='regression')
    engine.runExperiments(X, y, modelConfigs, numRepetitions=21)

if __name__ == "__main__":
    main()
