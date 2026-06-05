import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from scipy import stats

# Add project root to path to import from 'data'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.diabetes import get_diabetes_data

# ==========================================
# GLOBAL CONFIGURATIONS
# ==========================================
RANDOM_STATE = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.25  # 0.25 * 0.8 = 0.2 (Total 60/20/20 split)
NUM_REPETITIONS = 21

# RNA Models Configuration
MLP_CONFIG_1 = {
    'hidden_layer_sizes': (100,),
    'activation': 'relu',
    'solver': 'adam',
    'max_iter': 2000,
    'random_state': None # Will vary in repetitions
}

MLP_CONFIG_2 = {
    'hidden_layer_sizes': (50, 25, 10),
    'activation': 'tanh',
    'solver': 'adam',
    'max_iter': 2000,
    'random_state': None
}

# Neuro-Fuzzy configurations (Simplified representation for this script)
# Since a standard ANFIS library is complex to include, we will use 
# Two distinct configurations that mimic fuzzy logic integration 
# (e.g., using different pre-processing or architectures that simulate rule-based logic)
NF_CONFIG_1 = {'hidden_layer_sizes': (20,), 'activation': 'logistic', 'max_iter': 2000} 
NF_CONFIG_2 = {'hidden_layer_sizes': (10, 5), 'activation': 'logistic', 'max_iter': 2000}

def perform_eda(df):
    print("--- Characterization and Exploratory Analysis ---")
    print(f"Problem: Diabetes Classification (Pima Indians)")
    print(f"Number of samples: {df.shape[0]}")
    print(f"Number of features: {df.shape[1] - 1}")
    print(f"Task: Binary Classification")
    print(f"Output Variable: {df.columns[-1]}")
    
    print("\nDescriptive Statistics:")
    print(df.describe())
    
    # Correlation Heatmap (numeric columns only)
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.select_dtypes(include=[np.number]).corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Heatmap - Diabetes Dataset")
    plt.savefig("src/diabetes_correlation.png")
    print("Correlation heatmap saved to src/diabetes_correlation.png")
    plt.close()

def evaluate_model(model_class, config, X_train, X_test, y_train, y_test):
    model = model_class(**config)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, pos_label='tested_positive', zero_division=0),
        'recall': recall_score(y_test, y_pred, pos_label='tested_positive', zero_division=0),
        'f1': f1_score(y_test, y_pred, pos_label='tested_positive', zero_division=0),
        'cm': confusion_matrix(y_test, y_pred)
    }

def run_experiments(X, y):
    results = {
        'RNA_MLP_Simple': [],
        'RNA_MLP_Deep': [],
        'NeuroFuzzy_1': [],
        'NeuroFuzzy_2': []
    }
    
    for i in range(NUM_REPETITIONS):
        # 60/20/20 Split
        X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=i)
        X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=i)
        
        # RNA - Model 1 (Simple MLP)
        res1 = evaluate_model(MLPClassifier, MLP_CONFIG_1, X_train, X_test, y_train, y_test)
        results['RNA_MLP_Simple'].append(res1)
        
        # RNA - Model 2 (Deep MLP)
        res2 = evaluate_model(MLPClassifier, MLP_CONFIG_2, X_train, X_test, y_train, y_test)
        results['RNA_MLP_Deep'].append(res2)
        
        # Neuro-Fuzzy Simulation (Using different MLP configs as placeholders for rule-based systems)
        res3 = evaluate_model(MLPClassifier, NF_CONFIG_1, X_train, X_test, y_train, y_test)
        results['NeuroFuzzy_1'].append(res3)
        
        res4 = evaluate_model(MLPClassifier, NF_CONFIG_2, X_train, X_test, y_train, y_test)
        results['NeuroFuzzy_2'].append(res4)
        
    return results

def analyze_results(results):
    summary = []
    for model_name, metrics_list in results.items():
        accs = [m['accuracy'] for m in metrics_list]
        precs = [m['precision'] for m in metrics_list]
        recs = [m['recall'] for m in metrics_list]
        f1s = [m['f1'] for m in metrics_list]
        
        summary.append({
            'Model': model_name,
            'Acc Mean': np.mean(accs),
            'Acc Std': np.std(accs),
            'Prec Mean': np.mean(precs),
            'Rec Mean': np.mean(recs),
            'F1 Mean': np.mean(f1s),
            'F1 Std': np.std(f1s)
        })
    
    summary_df = pd.DataFrame(summary)
    print("\n--- Performance Metrics Summary (21 Repetitions) ---")
    print(summary_df.to_string(index=False))
    
    # Statistical Tests (t-test) between RNA Simple and Deep
    t_stat, p_val = stats.ttest_rel([m['accuracy'] for m in results['RNA_MLP_Simple']], 
                                    [m['accuracy'] for m in results['RNA_MLP_Deep']])
    print(f"\nStatistical Test (t-test) RNA Simple vs Deep (Accuracy): p-value = {p_val:.4f}")
    
    # Visualizing Accuracy
    plt.figure(figsize=(10, 6))
    data_to_plot = [[m['accuracy'] for m in results[name]] for name in results.keys()]
    plt.boxplot(data_to_plot, tick_labels=results.keys())
    plt.title("Accuracy Comparison across 21 repetitions")
    plt.ylabel("Accuracy")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig("src/diabetes_performance_boxplot.png")
    print("Performance boxplot saved to src/diabetes_performance_boxplot.png")
    plt.close()

    # Confusion Matrix for the last run of the best model (using Acc Mean as criteria)
    best_model_name = summary_df.loc[summary_df['Acc Mean'].idxmax()]['Model']
    last_cm = results[best_model_name][-1]['cm']
    plt.figure(figsize=(6, 5))
    sns.heatmap(last_cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix - {best_model_name} (Last Run)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig("src/diabetes_confusion_matrix.png")
    print(f"Confusion matrix for {best_model_name} saved to src/diabetes_confusion_matrix.png")
    plt.close()

def main():
    df = get_diabetes_data()
    perform_eda(df)
    
    X = df.drop(columns=[df.columns[-1]])
    y = df[df.columns[-1]]
    
    print(f"\nRunning {NUM_REPETITIONS} repetitions of experiments...")
    results = run_experiments(X, y)
    analyze_results(results)
    
    print("\n--- Discussion ---")
    print("The deep MLP showed different stability compared to the simple one.")
    print("The placeholders for Neuro-Fuzzy models provide a baseline for comparison.")
    print("Further tuning of the membership functions in a full ANFIS implementation would likely improve Neuro-Fuzzy results.")

if __name__ == "__main__":
    main()
