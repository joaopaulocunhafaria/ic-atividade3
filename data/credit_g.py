import pandas as pd
from sklearn.datasets import fetch_openml
import os

def get_credit_g_data():
    """Fetches and processes the Credit-g dataset."""
    # Fetch dataset
    credit_g = fetch_openml(name='credit-g', version=1, as_frame=True, parser='auto')
    
    df = credit_g.frame
    
    # 1. Handling Missing Values
    df = df.dropna()
    
    # 2. One-Hot Encoding for categorical features
    categorical_cols = df.select_dtypes(include=['category', 'object']).columns
    features_to_encode = [col for col in categorical_cols if col != 'class']
    df = pd.get_dummies(df, columns=features_to_encode)
    
    return df

if __name__ == "__main__":
    df = get_credit_g_data()
    print("Credit-g processed shape:", df.shape)
    print(df.head())
