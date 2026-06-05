import pandas as pd
from ucimlrepo import fetch_ucirepo
import os

def get_auto_mpg_data():
    """Fetches and processes the Auto MPG dataset."""
    # Fetch dataset
    auto_mpg = fetch_ucirepo(id=9)
    
    # Data (as pandas dataframes)
    X = auto_mpg.data.features
    y = auto_mpg.data.targets

    # Combine for treatment
    df = pd.concat([X, y], axis=1)
    
    # 1. Handling Missing Values
    if df.isnull().values.any():
        df = df.dropna()
    
    # 2. One-Hot Encoding for 'origin'
    df = pd.get_dummies(df, columns=['origin'], prefix='origin')
    
    return df

if __name__ == "__main__":
    df = get_auto_mpg_data()
    print("Auto MPG processed shape:", df.shape)
    print(df.head())
