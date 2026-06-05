import pandas as pd
from sklearn.datasets import fetch_openml
import os

def get_diabetes_data():
    """Fetches and processes the Diabetes dataset."""
    # Fetch dataset
    diabetes = fetch_openml(name='diabetes', version=1, as_frame=True, parser='auto')
    
    df = diabetes.frame
    
    # 1. Handling Missing Values
    df = df.dropna()
    
    return df

if __name__ == "__main__":
    df = get_diabetes_data()
    print("Diabetes processed shape:", df.shape)
    print(df.head())
