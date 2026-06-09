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

def performEda(df, outputDir):
    print("\n--- Exploratory Data Analysis: Credit-G ---")
    print(f"Samples: {df.shape[0]}, Features: {df.shape[1]-1}")
    
    # Class distribution
    plt.figure(figsize=(8, 6))
    sns.countplot(x=df.columns[-1], data=df)
    plt.title("Class Distribution - Credit-G")
    plt.savefig(f"{outputDir}/edaClassDist.png")
    plt.close()

def main():
    datasetName = "credit_g"
    outputDir = f"output/{datasetName}"
    os.makedirs(outputDir, exist_ok=True)
    
    # 1. Load Data
    df = get_credit_g_data()
    
    # 2. EDA
    performEda(df, outputDir)
    
    # 3. Prepare Features and Target
    X = df.drop(columns=['class'])
    y = df['class']
    
    # 4. Define Models
    models = {
        'RNA_MLP_Simple': {
            'class': MLPClassifier,
            'params': {'hidden_layer_sizes': (100,), 'activation': 'relu', 'max_iter': 2000, 'random_state': 42}
        },
        'RNA_MLP_Deep': {
            'class': MLPClassifier,
            'params': {'hidden_layer_sizes': (50, 25, 10), 'activation': 'tanh', 'max_iter': 2000, 'random_state': 42}
        },
        'NeuroFuzzy_TSK_3': {
            'class': TSKClassifier,
            'params': {'nClusters': 3, 'm': 2.0}
        },
        'NeuroFuzzy_TSK_5': {
            'class': TSKClassifier,
            'params': {'nClusters': 5, 'm': 2.0}
        }
    }
    
    # 5. Run Experiments
    engine = ExperimentEngine(datasetName, taskType='classification')
    engine.runExperiments(X, y, models, numRepetitions=21)

if __name__ == "__main__":
    main()
