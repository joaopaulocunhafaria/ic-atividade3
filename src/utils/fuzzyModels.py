import numpy as np
import skfuzzy as fuzz
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.linear_model import LinearRegression, LogisticRegression

class TSKModel(BaseEstimator):
    """
    Takagi-Sugeno-Kang (TSK) Fuzzy Inference System.
    Uses Fuzzy C-Means clustering to define antecedents and 
    linear models for consequents.
    """
    def __init__(self, nClusters=3, m=2.0, taskType='regression'):
        self.nClusters = nClusters
        self.m = m # Fuzziness exponent
        self.taskType = taskType
        self.clusterCenters = None
        self.consequentModels = []

    def fit(self, X, y):
        X = np.asanyarray(X)
        y = np.asanyarray(y)
        
        # 1. Clustering (Antecedents)
        # skfuzzy expects data as (features, samples)
        cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
            X.T, self.nClusters, self.m, error=0.005, maxiter=1000, init=None
        )
        self.clusterCenters = cntr
        
        # 2. Consequents
        # For each cluster, we train a local model weighted by the membership degree
        self.consequentModels = []
        for i in range(self.nClusters):
            weights = u[i, :]
            if self.taskType == 'regression':
                model = LinearRegression()
                model.fit(X, y, sample_weight=weights)
            else:
                model = LogisticRegression(max_iter=1000)
                # Map classes to 0, 1 if they are not
                model.fit(X, y, sample_weight=weights)
            self.consequentModels.append(model)
        
        return self

    def predict(self, X):
        X = np.asanyarray(X)
        
        # Calculate membership degrees for new data
        u, u0, d, jm, p, fpc = fuzz.cluster.cmeans_predict(
            X.T, self.clusterCenters, self.m, error=0.005, maxiter=1000
        )
        
        # Weighted average of consequent models
        outputs = []
        for i in range(self.nClusters):
            if self.taskType == 'regression':
                out = self.consequentModels[i].predict(X)
            else:
                # For classification, we use the probability of class 1
                out = self.consequentModels[i].predict_proba(X)[:, 1]
            outputs.append(out)
        
        outputs = np.array(outputs)
        # Final output is sum(u_i * y_i) / sum(u_i)
        # Note: cmeans_predict normalizes u such that sum(u_i) = 1
        finalOutput = np.sum(u * outputs, axis=0)
        
        if self.taskType == 'classification':
            # This is a bit simplified for multi-class, but works for binary
            return (finalOutput > 0.5).astype(int)
        return finalOutput

class TSKClassifier(TSKModel, ClassifierMixin):
    def __init__(self, nClusters=3, m=2.0):
        super().__init__(nClusters=nClusters, m=m, taskType='classification')
        self.classes_ = None

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        # Map labels to 0 and 1 for the TSK logic if it's binary
        # For simplicity, we assume binary for now or standard sklearn label encoding
        from sklearn.preprocessing import LabelEncoder
        self.labelEncoder = LabelEncoder()
        yEncoded = self.labelEncoder.fit_transform(y)
        super().fit(X, yEncoded)
        return self
    
    def predict(self, X):
        preds = super().predict(X)
        return self.labelEncoder.inverse_transform(preds)

class TSKRegressor(TSKModel, RegressorMixin):
    def __init__(self, nClusters=3, m=2.0):
        super().__init__(nClusters=nClusters, m=m, taskType='regression')
