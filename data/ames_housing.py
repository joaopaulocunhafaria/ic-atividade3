import pandas as pd
from sklearn.datasets import fetch_openml
import os

def get_ames_housing_data():
    """Fetches and processes the Ames Housing dataset."""
    # Fetch dataset
    ames = fetch_openml(name='house_prices', version=1, as_frame=True, parser='auto')
    
    df = ames.frame
    
    # 1. Handling Missing Values
    threshold = 0.4 * len(df)
    df = df.dropna(thresh=threshold, axis=1).copy()
    
    numeric_cols = df.select_dtypes(include=['number']).columns
    df.loc[:, numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    
    categorical_cols = df.select_dtypes(include=['category', 'object']).columns
    for col in categorical_cols:
        df.loc[:, col] = df[col].fillna(df[col].mode()[0])
    
    # 2. One-Hot Encoding
    df = pd.get_dummies(df, columns=categorical_cols)
    
    return df

if __name__ == "__main__":
    df = get_ames_housing_data()
    print("Ames Housing processed shape:", df.shape)
    print(df.head())
