import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neural_network import MLPClassifier

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.credit_g import get_credit_g_data
from src.experimentEngine import ExperimentEngine
from src.utils.fuzzyModels import TSKClassifier
from src.utils.rbfModels import RBFClassifier

def performEda(df, outputDir):
    """
    Performs Exploratory Data Analysis for the Credit-G dataset.
    """
    print("\n--- Exploratory Data Analysis: Credit-G ---")
    
    # Class distribution
    plt.figure(figsize=(8, 6))
    sns.countplot(x='class', data=df)
    plt.title("Class Distribution - Credit-G")
    plt.savefig(f"{outputDir}/edaClassDist.png")
    plt.close()
    
    # descriptive stats
    df.describe().to_csv(f"{outputDir}/descriptiveStats.csv")

def main():
    datasetName = "credit_g"
    outputDir = f"output/{datasetName}"
    os.makedirs(outputDir, exist_ok=True)
    
    # 1. Load Data
    df = get_credit_g_data()
    performEda(df, outputDir)
    
    # 2. Prepare Features and Target
    X = df.drop(columns=['class'])
    y = df['class']
    
    # 3. Define Model Configurations and Parameter Grids
    modelConfigs = {
        'MLP': {
            'class': MLPClassifier,
            'paramGrid': {
                'hidden_layer_sizes': [(100,), (50, 50)],
                'activation': ['relu', 'tanh'],
                'max_iter': [2000],
                'random_state': [42]
            }
        },
        'RBF': {
            'class': RBFClassifier,
            'paramGrid': {
                'nCenters': [10, 30, 50],
                'gamma': [0.1, 1.0]
            }
        },
        'TSK_Variation_1': {
            'class': TSKClassifier,
            'paramGrid': {
                'nClusters': [3, 5],
                'm': [2.0]
            }
        },
        'TSK_Variation_2': {
            'class': TSKClassifier,
            'paramGrid': {
                'nClusters': [8, 10],
                'm': [1.5, 2.0]
            }
        }
    }
    
    # 4. Run Systematic Experiments
    engine = ExperimentEngine(datasetName, taskType='classification')
    engine.runExperiments(X, y, modelConfigs, numRepetitions=21)

if __name__ == "__main__":
    main()
