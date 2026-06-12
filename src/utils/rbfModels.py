import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.linear_model import Ridge, RidgeClassifier

class RBFNetwork(BaseEstimator):
    """
    Radial Basis Function (RBF) Network.
    Uses KMeans for clustering centers and Ridge (Linear) for the output layer.
    """
    def __init__(self, nCenters=10, gamma=1.0, taskType='regression'):
        self.nCenters = nCenters
        self.gamma = gamma
        self.taskType = taskType
        self.kmeans = KMeans(n_clusters=nCenters, n_init='auto', random_state=42)
        
        if taskType == 'regression':
            self.outputLayer = Ridge()
        else:
            self.outputLayer = RidgeClassifier()

    def _getFeatures(self, X):
        # Calculate RBF activations: exp(-gamma * ||x - c||^2)
        return rbf_kernel(X, self.centers, gamma=self.gamma)

    def fit(self, X, y):
        # 1. Find centers using KMeans
        self.kmeans.fit(X)
        self.centers = self.kmeans.cluster_centers_
        
        # 2. Transform input to RBF space
        rbfFeatures = self._getFeatures(X)
        
        # 3. Train the linear output layer
        self.outputLayer.fit(rbfFeatures, y)
        return self

    def predict(self, X):
        rbfFeatures = self._getFeatures(X)
        return self.outputLayer.predict(rbfFeatures)

class RBFClassifier(RBFNetwork, ClassifierMixin):
    def __init__(self, nCenters=10, gamma=1.0):
        super().__init__(nCenters=nCenters, gamma=gamma, taskType='classification')

class RBFRegressor(RBFNetwork, RegressorMixin):
    def __init__(self, nCenters=10, gamma=1.0):
        super().__init__(nCenters=nCenters, gamma=gamma, taskType='regression')
