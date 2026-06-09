import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                             confusion_matrix, mean_squared_error, mean_absolute_error, r2_score)
from sklearn.preprocessing import StandardScaler
from scipy import stats
from sklearn.exceptions import ConvergenceWarning

# Suppress ConvergenceWarnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)

class ExperimentEngine:
    def __init__(self, datasetName, taskType='classification'):
        self.datasetName = datasetName
        self.taskType = taskType
        self.outputDir = f"output/{datasetName}"
        os.makedirs(self.outputDir, exist_ok=True)
        self.results = {}

    def runExperiments(self, X, y, models, numRepetitions=21):
        print(f"\n>>> Starting experiments for dataset: {self.datasetName} ({self.taskType})")
        
        for modelName, modelInfo in models.items():
            print(f"  Training model: {modelName}")
            modelClass = modelInfo['class']
            modelParams = modelInfo['params']
            modelResults = []

            for i in range(numRepetitions):
                # 60/20/20 Split
                # 80% train+val, 20% test
                X_trainVal, X_test, y_trainVal, y_test = train_test_split(
                    X, y, test_size=0.20, random_state=i
                )
                # From 80%, 25% is validation (which is 20% of total)
                X_train, X_val, y_train, y_val = train_test_split(
                    X_trainVal, y_trainVal, test_size=0.25, random_state=i
                )

                # Scaling
                scaler = StandardScaler()
                X_trainScaled = scaler.fit_transform(X_train)
                X_valScaled = scaler.transform(X_val)
                X_testScaled = scaler.transform(X_test)

                # Train model
                model = modelClass(**modelParams)
                model.fit(X_trainScaled, y_train)
                
                # Predict on test
                yPred = model.predict(X_testScaled)
                
                # Calculate metrics
                metrics = self._calculateMetrics(y_test, yPred)
                modelResults.append(metrics)

            self.results[modelName] = modelResults
        
        self._saveResults()
        self._generatePlots()

    def _calculateMetrics(self, yTrue, yPred):
        if self.taskType == 'classification':
            # Handle possible string labels by finding the 'positive' class if binary
            uniqueLabels = np.unique(yTrue)
            posLabel = uniqueLabels[1] if len(uniqueLabels) == 2 else uniqueLabels[0]
            
            return {
                'accuracy': accuracy_score(yTrue, yPred),
                'precision': precision_score(yTrue, yPred, pos_label=posLabel, average='binary', zero_division=0),
                'recall': recall_score(yTrue, yPred, pos_label=posLabel, average='binary', zero_division=0),
                'f1': f1_score(yTrue, yPred, pos_label=posLabel, average='binary', zero_division=0),
                'cm': confusion_matrix(yTrue, yPred)
            }
        else:
            mse = mean_squared_error(yTrue, yPred)
            return {
                'mse': mse,
                'rmse': np.sqrt(mse),
                'mae': mean_absolute_error(yTrue, yPred),
                'r2': r2_score(yTrue, yPred)
            }

    def _saveResults(self):
        summary = []
        for modelName, metricsList in self.results.items():
            dfMetrics = pd.DataFrame(metricsList)
            # Exclude confusion matrix from mean/std
            numericMetrics = dfMetrics.drop(columns=['cm']) if 'cm' in dfMetrics.columns else dfMetrics
            
            means = numericMetrics.mean()
            stds = numericMetrics.std()
            
            row = {'model': modelName}
            for col in numericMetrics.columns:
                row[f"{col}Mean"] = means[col]
                row[f"{col}Std"] = stds[col]
            summary.append(row)
        
        summaryDf = pd.DataFrame(summary)
        summaryDf.to_csv(f"{self.outputDir}/metricsSummary.csv", index=False)
        print(f"\nResults summary saved to {self.outputDir}/metricsSummary.csv")
        print(summaryDf.to_string(index=False))

    def _generatePlots(self):
        # 1. Boxplot of main metric
        mainMetric = 'accuracy' if self.taskType == 'classification' else 'r2'
        plt.figure(figsize=(10, 6))
        dataToPlot = [ [m[mainMetric] for m in self.results[name]] for name in self.results.keys()]
        plt.boxplot(dataToPlot, tick_labels=list(self.results.keys()))
        plt.title(f"{mainMetric.capitalize()} Comparison - {self.datasetName}")
        plt.ylabel(mainMetric.capitalize())
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(f"{self.outputDir}/performanceBoxplot.png")
        plt.close()

        # 2. Confusion Matrix for best model (classification only)
        if self.taskType == 'classification':
            summaryDf = pd.read_csv(f"{self.outputDir}/metricsSummary.csv")
            bestModelName = summaryDf.loc[summaryDf['accuracyMean'].idxmax()]['model']
            # Get CM from last run of best model
            lastCm = self.results[bestModelName][-1]['cm']
            plt.figure(figsize=(6, 5))
            sns.heatmap(lastCm, annot=True, fmt='d', cmap='Blues')
            plt.title(f"Confusion Matrix - {bestModelName} (Last Run)")
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.savefig(f"{self.outputDir}/confusionMatrix.png")
            plt.close()
