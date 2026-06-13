import numpy as np
import skfuzzy as fuzz
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.linear_model import LinearRegression, LogisticRegression

class TSKModel(BaseEstimator):
    """
    Takagi-Sugeno-Kang (TSK) Fuzzy Inference System implementation.
    
    This model uses Fuzzy C-Means (FCM) clustering to define the antecedents (membership functions)
    and trains local linear models for the consequents of each fuzzy rule.
    The final output is a weighted average of the local models based on membership degrees.
    """
    def __init__(self, nClusters=3, m=2.0, taskType='regression'):
        self.nClusters = nClusters
        self.m = m # Fuzziness exponent (typically 2.0)
        self.taskType = taskType
        self.clusterCenters = None
        self.consequentModels = []

    def fit(self, X, y):
        """
        Fits the TSK model by identifying clusters and training local consequent models.
        """
        X = np.asanyarray(X)
        y = np.asanyarray(y)
        
        # 1. Clustering Phase (Defining Antecedents)
        # skfuzzy expects data shape: (features, samples)
        cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
            X.T, self.nClusters, self.m, error=0.005, maxiter=1000, init=None
        )
        self.clusterCenters = cntr
        
        # 2. Consequent Training Phase
        # For each fuzzy rule (cluster), we train a local model weighted by membership
        self.consequentModels = []
        for i in range(self.nClusters):
            weights = u[i, :]
            if self.taskType == 'regression':
                model = LinearRegression()
                model.fit(X, y, sample_weight=weights)
            else:
                model = LogisticRegression(max_iter=1000)
                model.fit(X, y, sample_weight=weights)
            self.consequentModels.append(model)
        
        return self

    def predict(self, X):
        """
        Predicts using the fuzzy inference logic: sum(membership_i * output_i).
        """
        X = np.asanyarray(X)
        
        # Calculate membership degrees for new data points
        u, u0, d, jm, p, fpc = fuzz.cluster.cmeans_predict(
            X.T, self.clusterCenters, self.m, error=0.005, maxiter=1000
        )
        
        # Aggregate local model predictions
        outputs = []
        for i in range(self.nClusters):
            if self.taskType == 'regression':
                out = self.consequentModels[i].predict(X)
            else:
                # Currently optimized for binary classification
                out = self.consequentModels[i].predict_proba(X)[:, 1]
            outputs.append(out)
        
        outputs = np.array(outputs)
        # Membership degrees (u) are already normalized by cmeans_predict
        finalOutput = np.sum(u * outputs, axis=0)
        
        if self.taskType == 'classification':
            return (finalOutput > 0.5).astype(int)
        return finalOutput

class TSKClassifier(TSKModel, ClassifierMixin):
    """
    TSK Fuzzy Classifier wrapper for scikit-learn compatibility.
    """
    def __init__(self, nClusters=3, m=2.0):
        super().__init__(nClusters=nClusters, m=m, taskType='classification')
        self.classes_ = None

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        from sklearn.preprocessing import LabelEncoder
        self.labelEncoder = LabelEncoder()
        yEncoded = self.labelEncoder.fit_transform(y)
        super().fit(X, yEncoded)
        return self
    
    def predict(self, X):
        preds = super().predict(X)
        return self.labelEncoder.inverse_transform(preds)

class TSKRegressor(TSKModel, RegressorMixin):
    """
    TSK Fuzzy Regressor wrapper for scikit-learn compatibility.
    """
    def __init__(self, nClusters=3, m=2.0):
        super().__init__(nClusters=nClusters, m=m, taskType='regression')
