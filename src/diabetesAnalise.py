import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neural_network import MLPClassifier

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.diabetes import get_diabetes_data
from src.experimentEngine import ExperimentEngine
from src.utils.fuzzyModels import TSKClassifier
from src.utils.rbfModels import RBFClassifier
from src.utils.dataBaseInfo import DatasetInfoGenerator

def performEda(df, outputDir):
    """
    Performs basic Exploratory Data Analysis.
    Saves descriptive statistics and a correlation heatmap.
    """
    print("\n--- Exploratory Data Analysis: Diabetes ---")
    
    # Save descriptive stats
    df.describe().to_csv(f"{outputDir}/descriptiveStats.csv")
    
    # Correlation Heatmap
    plt.figure(figsize=(10, 8))
    numericDf = df.select_dtypes(include=[np.number])
    sns.heatmap(numericDf.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Heatmap - Diabetes")
    plt.savefig(f"{outputDir}/edaCorrelation.png")
    plt.close()

def main():
    datasetName = "diabetes"
    outputDir = f"output/{datasetName}"
    os.makedirs(outputDir, exist_ok=True)
    
    # 1. Load and process data
    df = get_diabetes_data()
    performEda(df, outputDir)
    description = "Classificação de pacientes quanto à presença ou não de diabetes com base em medidas diagnósticas."
    DatasetInfoGenerator.generateDatasetInfo(df, taskType='classification', outputDir=outputDir, description=description)
    
    X = df.drop(columns=[df.columns[-1]])
    y = df[df.columns[-1]]
    
    # 2. Define Model Configurations and Parameter Grids
    modelConfigs = {
        'MLP': {
            'class': MLPClassifier,
            'paramGrid': {
                'hidden_layer_sizes': [(50,), (100,), (50, 25), (100, 50)],
                'activation': ['relu', 'tanh'],
                'solver': ['adam', 'sgd'],
                'alpha': [0.0001, 0.01],
                'learning_rate_init': [0.001, 0.01],
                'max_iter': [1500]
            }
        },
        'RBF': {
            'class': RBFClassifier,
            'paramGrid': {
                'nCenters': [5, 10, 20, 30],
                'gamma': [0.01, 0.1, 1.0, 10.0]
            }
        },
        'TSK_Variation_1': {
            'class': TSKClassifier,
            'paramGrid': {
                'nClusters': [2, 3, 5],
                'm': [1.5, 2.0]
            }
        },
        'TSK_Variation_2': {
            'class': TSKClassifier,
            'paramGrid': {
                'nClusters': [7, 10],
                'm': [2.0, 2.5]
            }
        }
    }
    
    # 3. Run Systematic Experiments
    engine = ExperimentEngine(datasetName, taskType='classification')
    engine.runExperiments(X, y, modelConfigs, numRepetitions=21)

if __name__ == "__main__":
    main()
