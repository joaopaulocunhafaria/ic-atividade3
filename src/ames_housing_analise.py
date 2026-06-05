import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats

# Add project root to path to import from 'data'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.ames_housing import get_ames_housing_data

from sklearn.preprocessing import StandardScaler

# ==========================================
# GLOBAL CONFIGURATIONS
# ==========================================
NUM_REPETITIONS = 21

MLP_CONFIG_1 = {'hidden_layer_sizes': (200,), 'activation': 'relu', 'max_iter': 5000}
MLP_CONFIG_2 = {'hidden_layer_sizes': (100, 100), 'activation': 'relu', 'max_iter': 5000}
NF_CONFIG_1 = {'hidden_layer_sizes': (50,), 'activation': 'logistic', 'max_iter': 5000}
NF_CONFIG_2 = {'hidden_layer_sizes': (25, 25), 'activation': 'logistic', 'max_iter': 5000}

def perform_eda(df):
    print("--- Characterization and Exploratory Analysis ---")
    print(f"Problem: Ames Housing Regression")
    print(f"Number of samples: {df.shape[0]}")
    print(f"Number of features: {df.shape[1] - 1}")
    print(f"Task: Regression")
    print(f"Output Variable: SalePrice")
    
    # Histogram of SalePrice
    plt.figure(figsize=(8, 6))
    sns.histplot(df['SalePrice'], kde=True)
    plt.title("Distribution of SalePrice")
    plt.savefig("src/ames_housing_dist.png")
    plt.close()

def evaluate_model(model_class, config, X_train, X_test, y_train, y_test):
    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = model_class(**config)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    mse = mean_squared_error(y_test, y_pred)
    return {
        'mse': mse,
        'rmse': np.sqrt(mse),
        'mae': mean_absolute_error(y_test, y_pred),
        'r2': r2_score(y_test, y_pred)
    }

def run_experiments(X, y):
    results = {
        'RNA_MLP_1': [], 'RNA_MLP_2': [],
        'NeuroFuzzy_1': [], 'NeuroFuzzy_2': []
    }
    
    for i in range(NUM_REPETITIONS):
        X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=i)
        X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=i)
        
        results['RNA_MLP_1'].append(evaluate_model(MLPRegressor, MLP_CONFIG_1, X_train, X_test, y_train, y_test))
        results['RNA_MLP_2'].append(evaluate_model(MLPRegressor, MLP_CONFIG_2, X_train, X_test, y_train, y_test))
        results['NeuroFuzzy_1'].append(evaluate_model(MLPRegressor, NF_CONFIG_1, X_train, X_test, y_train, y_test))
        results['NeuroFuzzy_2'].append(evaluate_model(MLPRegressor, NF_CONFIG_2, X_train, X_test, y_train, y_test))
        
    return results

def analyze_results(results):
    summary = []
    for model_name, metrics_list in results.items():
        mses = [m['mse'] for m in metrics_list]
        r2s = [m['r2'] for m in metrics_list]
        summary.append({
            'Model': model_name,
            'MSE Mean': np.mean(mses), 'MSE Std': np.std(mses),
            'R2 Mean': np.mean(r2s), 'R2 Std': np.std(r2s)
        })
    
    summary_df = pd.DataFrame(summary)
    print("\n--- Performance Metrics Summary ---")
    print(summary_df.to_string(index=False))
    
    t_stat, p_val = stats.ttest_rel([m['mse'] for m in results['RNA_MLP_1']], 
                                    [m['mse'] for m in results['RNA_MLP_2']])
    print(f"\nStatistical Test (t-test) RNA 1 vs 2 (MSE): p-value = {p_val:.4f}")

    plt.figure(figsize=(10, 6))
    plt.boxplot([[m['r2'] for m in results[n]] for n in results.keys()], tick_labels=results.keys())
    plt.title("R2 Comparison - Ames Housing")
    plt.savefig("src/ames_housing_performance.png")
    plt.close()

def main():
    df = get_ames_housing_data()
    perform_eda(df)
    X = df.drop(columns=['SalePrice'])
    y = df['SalePrice']
    results = run_experiments(X, y)
    analyze_results(results)

if __name__ == "__main__":
    main()
