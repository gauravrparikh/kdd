import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.cluster import MiniBatchKMeans
from .engine import Clis

class ClisForest(BaseEstimator, ClusterMixin):
    def __init__(
        self, 
        n_estimators=10, 
        bootstrap_sample_ratio=0.5,
        n_clusters=3,
        random_state=42,
        **tree_params
    ):
        self.n_estimators = n_estimators
        self.bootstrap_sample_ratio = bootstrap_sample_ratio
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.tree_params = tree_params
        self.trees = []

    def fit(self, X, y):
        np.random.seed(self.random_state)
        self.trees = []
        n_samples = len(X)
        sample_size = int(n_samples * self.bootstrap_sample_ratio)

        for i in range(self.n_estimators):
            indices = np.random.choice(n_samples, sample_size, replace=True)
            X_sample = X.iloc[indices]
            y_sample = y[indices]

            tree = Clis(random_state=self.random_state + i, **self.tree_params)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
        
        return self

    def predict(self, X):
        """
        Modified Scalable Consensus:
        Uses Leaf-Feature Embedding + MiniBatchKMeans instead of 
        an N x N Co-association matrix.
        """
        n_samples = len(X)
        
        # 1. Generate an 'Embedding' of leaf IDs
        # Shape: (n_samples, n_estimators)
        leaf_matrix = np.zeros((n_samples, self.n_estimators), dtype=int)
        
        print(f"Generating leaf assignments from {self.n_estimators} trees...")
        for i, tree in enumerate(self.trees):
            leaf_matrix[:, i] = tree.predict(X)
        
        # 2. Perform Consensus via MiniBatchKMeans on the leaf assignments
        # This treats the leaf IDs as features. 
        # Note: Since leaf IDs are categorical, we use one-hot encoding or 
        # a high-speed partitioner to find commonalities.
        print(f"Performing Scalable Consensus for {self.n_clusters} clusters...")
        
        # We use MiniBatchKMeans because it scales O(N) rather than O(N^3)
        consensus_model = MiniBatchKMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            batch_size=1024,
            n_init="auto"
        )
        
        # Final prediction
        return consensus_model.fit_predict(leaf_matrix)