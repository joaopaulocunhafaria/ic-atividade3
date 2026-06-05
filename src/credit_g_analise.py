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
from data.credit_g import get_credit_g_data

# ==========================================
# GLOBAL CONFIGURATIONS
# ==========================================
TEST_SIZE = 0.2
VAL_SIZE = 0.25
NUM_REPETITIONS = 21

MLP_CONFIG_1 = {'hidden_layer_sizes': (100,), 'activation': 'relu', 'max_iter': 1000}
MLP_CONFIG_2 = {'hidden_layer_sizes': (100, 50), 'activation': 'tanh', 'max_iter': 1000}
NF_CONFIG_1 = {'hidden_layer_sizes': (30,), 'activation': 'logistic', 'max_iter': 1000}
NF_CONFIG_2 = {'hidden_layer_sizes': (15, 10), 'activation': 'logistic', 'max_iter': 1000}

def perform_eda(df):
    print("--- Characterization and Exploratory Analysis ---")
    print(f"Problem: German Credit Classification (Credit-g)")
    print(f"Number of samples: {df.shape[0]}")
    print(f"Number of features: {df.shape[1] - 1}")
    print(f"Task: Binary Classification")
    print(f"Output Variable: class")
    
    # Target distribution
    plt.figure(figsize=(6, 4))
    sns.countplot(x='class', data=df)
    plt.title("Target Distribution - Credit-g")
    plt.savefig("src/credit_g_distribution.png")
    plt.close()

def evaluate_model(model_class, config, X_train, X_test, y_train, y_test):
    model = model_class(**config)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, pos_label='good', zero_division=0),
        'recall': recall_score(y_test, y_pred, pos_label='good', zero_division=0),
        'f1': f1_score(y_test, y_pred, pos_label='good', zero_division=0),
        'cm': confusion_matrix(y_test, y_pred)
    }

def run_experiments(X, y):
    results = {
        'RNA_MLP_1': [], 'RNA_MLP_2': [],
        'NeuroFuzzy_1': [], 'NeuroFuzzy_2': []
    }
    
    for i in range(NUM_REPETITIONS):
        X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=i)
        X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=i)
        
        results['RNA_MLP_1'].append(evaluate_model(MLPClassifier, MLP_CONFIG_1, X_train, X_test, y_train, y_test))
        results['RNA_MLP_2'].append(evaluate_model(MLPClassifier, MLP_CONFIG_2, X_train, X_test, y_train, y_test))
        results['NeuroFuzzy_1'].append(evaluate_model(MLPClassifier, NF_CONFIG_1, X_train, X_test, y_train, y_test))
        results['NeuroFuzzy_2'].append(evaluate_model(MLPClassifier, NF_CONFIG_2, X_train, X_test, y_train, y_test))
        
    return results

def analyze_results(results):
    summary = []
    for model_name, metrics_list in results.items():
        accs = [m['accuracy'] for m in metrics_list]
        f1s = [m['f1'] for m in metrics_list]
        summary.append({
            'Model': model_name,
            'Acc Mean': np.mean(accs), 'Acc Std': np.std(accs),
            'F1 Mean': np.mean(f1s), 'F1 Std': np.std(f1s)
        })
    
    summary_df = pd.DataFrame(summary)
    print("\n--- Performance Metrics Summary ---")
    print(summary_df.to_string(index=False))
    
    # Stats test
    t_stat, p_val = stats.ttest_rel([m['accuracy'] for m in results['RNA_MLP_1']], 
                                    [m['accuracy'] for m in results['RNA_MLP_2']])
    print(f"\nStatistical Test (t-test) RNA 1 vs 2: p-value = {p_val:.4f}")

    plt.figure(figsize=(10, 6))
    plt.boxplot([[m['accuracy'] for m in results[n]] for n in results.keys()], tick_labels=results.keys())
    plt.title("Accuracy Comparison - Credit-g")
    plt.savefig("src/credit_g_performance.png")
    plt.close()

def main():
    df = get_credit_g_data()
    perform_eda(df)
    X = df.drop(columns=['class'])
    y = df['class']
    results = run_experiments(X, y)
    analyze_results(results)

if __name__ == "__main__":
    main()
