import os
import pandas as pd
import numpy as np
from scipy.stats import friedmanchisquare

def runFinalComparison():
    """
    Aggregates results from all datasets and performs global statistical analysis.
    """
    datasets = ["ames_housing", "auto_mpg", "credit_g", "diabetes"]
    standardModels = ["MLP", "RBF", "TSK_Variation_1", "TSK_Variation_2"]
    outputBaseDir = "output/finalComparison"
    os.makedirs(outputBaseDir, exist_ok=True)
    
    allResults = []
    
    for ds in datasets:
        summaryPath = f"output/{ds}/metricsSummary.csv"
        if os.path.exists(summaryPath):
            df = pd.read_csv(summaryPath)
            # Determine main metric based on task type
            mainMetric = "r2Mean" if ds in ["ames_housing", "auto_mpg"] else "accuracyMean"
            
            for _, row in df.iterrows():
                modelName = row["model"]
                # Only include our current experimental models
                if modelName in standardModels:
                    allResults.append({
                        "dataset": ds,
                        "model": modelName,
                        "score": row[mainMetric]
                    })
    
    if not allResults:
        print("No results found to compare.")
        return
        
    resultsDf = pd.pivot_table(pd.DataFrame(allResults), values='score', index='dataset', columns='model')
    
    # Drop any model that doesn't have results for all datasets (Friedman requirement)
    resultsDf = resultsDf.dropna(axis=1)
    
    resultsDf.to_csv(f"{outputBaseDir}/globalComparisonMatrix.csv")
    print("\n--- Global Comparison Matrix ---")
    print(resultsDf)
    
    # Friedman Test requires at least 3 models and 2 datasets
    if resultsDf.shape[1] >= 3 and resultsDf.shape[0] >= 2:
        data = [resultsDf[col].values for col in resultsDf.columns]
        stat, pValue = friedmanchisquare(*data)
        
        with open(f"{outputBaseDir}/globalStatisticalTest.txt", "w", encoding="utf-8") as f:
            f.write("=== Friedman Test Results (Across all datasets) ===\n")
            f.write(f"Algorithms compared: {list(resultsDf.columns)}\n")
            f.write(f"Statistic: {stat:.4f}\n")
            f.write(f"P-value: {pValue:.4f}\n")
            f.write("Significant: " + ("Yes" if pValue < 0.05 else "No") + "\n")
        
        print(f"\nFriedman Test P-value: {pValue:.4f}")
    else:
        print("\nNot enough data for Friedman test (need at least 3 algorithms present in all datasets).")

if __name__ == "__main__":
    runFinalComparison()
