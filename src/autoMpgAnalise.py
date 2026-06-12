import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neural_network import MLPRegressor

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.auto_mpg import get_auto_mpg_data
from src.experimentEngine import ExperimentEngine
from src.utils.fuzzyModels import TSKRegressor
from src.utils.rbfModels import RBFRegressor

def performEda(df, outputDir):
    """
    Performs basic Exploratory Data Analysis.
    Saves descriptive statistics and a correlation heatmap.
    """
    print("\n--- Exploratory Data Analysis: Auto MPG ---")
    
    # descriptive stats
    df.describe().to_csv(f"{outputDir}/descriptiveStats.csv")
    
    # Correlation Heatmap
    plt.figure(figsize=(10, 8))
    numericDf = df.select_dtypes(include=[np.number])
    sns.heatmap(numericDf.corr(), annot=True, cmap='viridis', fmt=".2f")
    plt.title("Correlation Heatmap - Auto MPG")
    plt.savefig(f"{outputDir}/edaCorrelation.png")
    plt.close()

def main():
    datasetName = "auto_mpg"
    outputDir = f"output/{datasetName}"
    os.makedirs(outputDir, exist_ok=True)
    
    # 1. Load Data
    df = get_auto_mpg_data()
    performEda(df, outputDir)
    
    # 2. Prepare Features and Target
    X = df.drop(columns=['mpg'])
    y = df['mpg']
    
    # 3. Define Model Configurations and Parameter Grids
    modelConfigs = {
        'MLP': {
            'class': MLPRegressor,
            'paramGrid': {
                'hidden_layer_sizes': [(50,), (100,)],
                'activation': ['relu', 'tanh'],
                'max_iter': [3000],
                'random_state': [42]
            }
        },
        'RBF': {
            'class': RBFRegressor,
            'paramGrid': {
                'nCenters': [5, 15, 30],
                'gamma': [0.1, 1.0, 5.0]
            }
        },
        'TSK_Variation_1': {
            'class': TSKRegressor,
            'paramGrid': {
                'nClusters': [3, 5],
                'm': [2.0]
            }
        },
        'TSK_Variation_2': {
            'class': TSKRegressor,
            'paramGrid': {
                'nClusters': [8, 12],
                'm': [1.5, 2.0]
            }
        }
    }
    
    # 4. Run Systematic Experiments
    engine = ExperimentEngine(datasetName, taskType='regression')
    engine.runExperiments(X, y, modelConfigs, numRepetitions=21)

if __name__ == "__main__":
    main()
