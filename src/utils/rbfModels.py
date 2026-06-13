import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.linear_model import Ridge, RidgeClassifier

class RBFNetwork(BaseEstimator):
    """
    Radial Basis Function (RBF) Network implementation.
    
    This model uses KMeans to identify radial centers in the input space and then 
    transforms the input into a high-dimensional space based on the RBF kernel.
    A linear output layer (Ridge) is then trained on these transformed features.
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
        """
        Transforms input X into RBF features using the centers found during fitting.
        Formula: exp(-gamma * ||x - center||^2)
        """
        return rbf_kernel(X, self.centers, gamma=self.gamma)

    def fit(self, X, y):
        """
        Fits the RBF network: 
        1. Finds centers using KMeans.
        2. Transforms input to RBF space.
        3. Trains the linear output layer.
        """
        # 1. Identify radial centers
        self.kmeans.fit(X)
        self.centers = self.kmeans.cluster_centers_
        
        # 2. Project data onto RBF space
        rbfFeatures = self._getFeatures(X)
        
        # 3. Train linear mapping
        self.outputLayer.fit(rbfFeatures, y)
        return self

    def predict(self, X):
        """
        Predicts by projecting new data and passing it through the output layer.
        """
        rbfFeatures = self._getFeatures(X)
        return self.outputLayer.predict(rbfFeatures)

class RBFClassifier(RBFNetwork, ClassifierMixin):
    """
    RBF Network Classifier wrapper for scikit-learn compatibility.
    """
    def __init__(self, nCenters=10, gamma=1.0):
        super().__init__(nCenters=nCenters, gamma=gamma, taskType='classification')

class RBFRegressor(RBFNetwork, RegressorMixin):
    """
    RBF Network Regressor wrapper for scikit-learn compatibility.
    """
    def __init__(self, nCenters=10, gamma=1.0):
        super().__init__(nCenters=nCenters, gamma=gamma, taskType='regression')
