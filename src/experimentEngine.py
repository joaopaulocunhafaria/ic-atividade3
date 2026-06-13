import os
import itertools
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from scipy.stats import wilcoxon
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                             confusion_matrix, mean_squared_error, mean_absolute_error, r2_score)
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning


# Suppress warnings that could clutter the output
warnings.filterwarnings("ignore", category=ConvergenceWarning)

class ExperimentEngine:
    """
    Core engine for running systematic experiments.
    Handles data splitting, hyperparameter tuning via validation set, 
    repetitive executions, and result reporting.
    """
    def __init__(self, datasetName, taskType='classification'):
        self.datasetName = datasetName
        self.taskType = taskType
        self.outputDir = f"output/{datasetName}"
        os.makedirs(self.outputDir, exist_ok=True)
        self.results = {}

    def runExperiments(self, X, y, modelConfigs, numRepetitions=21):
        """
        Executes the experimental protocol for all provided models.
        """
        print(f"\n>>> Starting experiments: {self.datasetName} ({self.taskType})")
        
        for modelName, config in modelConfigs.items():
            print(f"  Evaluating Model: {modelName}")
            modelClass = config['class']
            paramGrid = config['paramGrid']
            modelResults = []

            for i in range(numRepetitions):
                # 1. Systematic Data Split (60/20/20)
                X_trainVal, X_test, y_trainVal, y_test = train_test_split(
                    X, y, test_size=0.20, random_state=i
                )
                X_train, X_val, y_train, y_val = train_test_split(
                    X_trainVal, y_trainVal, test_size=0.25, random_state=i
                )

                # 2. Scaling (Features and Target for Regression)
                scalerX = StandardScaler()
                X_trainScaled = scalerX.fit_transform(X_train)
                X_valScaled = scalerX.transform(X_val)
                X_testScaled = scalerX.transform(X_test)
                
                # MLP and other models often need the target scaled in regression to avoid overflow
                yTrainProcessed = y_train
                yValProcessed = y_val
                scalerY = None
                
                if self.taskType == 'regression':
                    scalerY = StandardScaler()
                    # Reshape to 2D for sklearn scaler
                    yTrainProcessed = scalerY.fit_transform(y_train.values.reshape(-1, 1)).flatten()
                    yValProcessed = scalerY.transform(y_val.values.reshape(-1, 1)).flatten()

                # 3. Hyperparameter Tuning on Validation Set
                bestParams = self._findBestParams(X_trainScaled, yTrainProcessed, X_valScaled, yValProcessed, modelClass, paramGrid)
                with open(f"{self.outputDir}/bestParametersLog.txt", "a", encoding="utf-8") as paramFile:
                    paramFile.write(f"Repetition {i+1} | Model: {modelName} | Params: {bestParams}\n")

                # 4. Final Training and Test Evaluation
                # Combining train and validation for the final model of this repetition
                X_finalTrain = np.vstack((X_trainScaled, X_valScaled))
                y_finalTrain = np.concatenate((yTrainProcessed, yValProcessed))
                
                finalModel = modelClass(**bestParams)
                startTime = time.time()
                finalModel.fit(X_finalTrain, y_finalTrain)
                trainingTime = time.time() - startTime
                
                yPredRaw = finalModel.predict(X_testScaled)
                
                # Inverse transform target if scaled, to calculate metrics in original units
                if scalerY:
                    yPred = scalerY.inverse_transform(yPredRaw.reshape(-1, 1)).flatten()
                else:
                    yPred = yPredRaw
                    
                metrics = self._calculateMetrics(y_test, yPred)
                metrics['trainingTimeSec'] = trainingTime
                modelResults.append(metrics)

            self.results[modelName] = modelResults
        
        self._summarizeAndSave()
        self._generateVisualizations()
        self._runStatisticalTests()
    
    def _runStatisticalTests(self):
        """
        Applies the Wilcoxon signed-rank test comparing the best algorithm with the others.
        This is a non-parametric paired test suitable for comparing model performance distributions.
        """
        print(f"\n--- Statistical Analysis (Wilcoxon) for {self.datasetName} ---")
        mainMetric = 'accuracy' if self.taskType == 'classification' else 'r2'
        
        # Extract distributions of the main metric for all models
        metricDistributions = {model: [res[mainMetric] for res in results] for model, results in self.results.items()}
        
        # Identify the best model (highest mean)
        bestModelName = max(metricDistributions, key=lambda m: np.mean(metricDistributions[m]))
        bestModelData = metricDistributions[bestModelName]
        
        statResults = []
        for modelName, data in metricDistributions.items():
            if modelName != bestModelName:
                # Check if distributions are identical to avoid zero-difference error in Wilcoxon
                if np.array_equal(bestModelData, data):
                    pValue = 1.0
                else:
                    try:
                        stat, pValue = wilcoxon(bestModelData, data)
                    except ValueError:
                        pValue = 1.0 # Fallback if test cannot be computed
                
                isSignificant = "Yes" if pValue < 0.05 else "No"
                statResults.append({
                    'BestModel': bestModelName,
                    'ComparedModel': modelName,
                    'p-value': pValue,
                    'SignificantDifference(5%)': isSignificant
                })
        
        if statResults:
            statDf = pd.DataFrame(statResults)
            statDf.to_csv(f"{self.outputDir}/statisticalTests.csv", index=False)
            print(statDf.to_string(index=False))
        else:
            print("Only one model evaluated; skipping statistical comparison.")

    def _findBestParams(self, X_train, y_train, X_val, y_val, modelClass, paramGrid):
        """Simple Grid Search logic using the validation set."""
        keys, values = zip(*paramGrid.items())
        combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
        
        bestScore = -np.inf
        bestParams = combinations[0]

        for params in combinations:
            model = modelClass(**params)
            model.fit(X_train, y_train)
            yValPred = model.predict(X_val)
            
            score = accuracy_score(y_val, yValPred) if self.taskType == 'classification' else r2_score(y_val, yValPred)
            if score > bestScore:
                bestScore = score
                bestParams = params
        
        return bestParams

    def _calculateMetrics(self, yTrue, yPred):
        """Calculates all metrics requested in the IEEE standard report."""
        if self.taskType == 'classification':
            labels = np.unique(yTrue)
            isBinary = len(labels) == 2
            method = 'binary' if isBinary else 'macro'
            pos = labels[1] if isBinary else None
            
            return {
                'accuracy': accuracy_score(yTrue, yPred),
                'precision': precision_score(yTrue, yPred, pos_label=pos, average=method, zero_division=0),
                'recall': recall_score(yTrue, yPred, pos_label=pos, average=method, zero_division=0),
                'f1': f1_score(yTrue, yPred, pos_label=pos, average=method, zero_division=0),
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

    def _summarizeAndSave(self):
        """Saves mean and std of metrics to a CSV file."""
        summary = []
        for modelName, metrics in self.results.items():
            df = pd.DataFrame(metrics)
            numericDf = df.drop(columns=['cm']) if 'cm' in df.columns else df
            
            row = {'model': modelName}
            for col in numericDf.columns:
                row[f"{col}Mean"] = numericDf[col].mean()
                row[f"{col}Std"] = numericDf[col].std()
            summary.append(row)
        
        summaryDf = pd.DataFrame(summary)
        summaryDf.to_csv(f"{self.outputDir}/metricsSummary.csv", index=False)
        print(f"\nResults summary saved to {self.outputDir}/metricsSummary.csv")
        print(summaryDf.to_string(index=False))

    def _generateVisualizations(self):
        """Generates plots for performance comparison."""
        mainMetric = 'accuracy' if self.taskType == 'classification' else 'r2'
        
        # 1. Performance Boxplot
        plt.figure(figsize=(10, 6))
        data = [[m[mainMetric] for m in self.results[name]] for name in self.results.keys()]
        plt.boxplot(data, tick_labels=list(self.results.keys()))
        plt.title(f"{mainMetric.capitalize()} Stability - {self.datasetName}")
        plt.ylabel(mainMetric.capitalize())
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.savefig(f"{self.outputDir}/performanceBoxplot.png")
        plt.close()

        # 2. Best Model Confusion Matrix
        if self.taskType == 'classification':
            summaryDf = pd.read_csv(f"{self.outputDir}/metricsSummary.csv")
            bestModel = summaryDf.loc[summaryDf['accuracyMean'].idxmax()]['model']
            lastCm = self.results[bestModel][-1]['cm']
            
            plt.figure(figsize=(6, 5))
            sns.heatmap(lastCm, annot=True, fmt='d', cmap='Blues')
            plt.title(f"Confusion Matrix (Best: {bestModel})")
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.savefig(f"{self.outputDir}/confusionMatrix.png")
            plt.close()
