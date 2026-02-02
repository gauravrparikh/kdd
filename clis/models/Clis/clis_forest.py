import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.cluster import SpectralClustering
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
        """
        Ensemble of CLIS trees using a Co-association Matrix for consensus.
        
        :param n_estimators: Number of CLIS trees to grow.
        :param bootstrap_sample_ratio: Fraction of data to sample for each tree.
        :param n_clusters: The final number of clusters for the consensus step.
        :param tree_params: Parameters passed to the individual Clis trees (loss_metric, complexity_penalty, etc).
        """
        self.n_estimators = n_estimators
        self.bootstrap_sample_ratio = bootstrap_sample_ratio
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.tree_params = tree_params
        self.trees = []

    def fit(self, X, y):
        """
        Build the forest by training multiple CLIS trees on bootstrap samples.
        """
        np.random.seed(self.random_state)
        self.trees = []
        n_samples = len(X)
        sample_size = int(n_samples * self.bootstrap_sample_ratio)

        for i in range(self.n_estimators):
            # Bootstrap sampling: random selection with replacement
            indices = np.random.choice(n_samples, sample_size, replace=True)
            X_sample = X.iloc[indices]
            y_sample = y[indices]

            # Initialize and fit individual CLIS tree
            tree = Clis(random_state=self.random_state + i, **self.tree_params)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
        
        return self

    def predict(self, X):
        """
        Consensus clustering using a Co-association Matrix.
        Points that frequently end up in the same leaf across the forest 
        are clustered together.
        """
        n_samples = len(X)
        # Initialize similarity matrix
        co_association = np.zeros((n_samples, n_samples))
        
        print(f"Building Co-association matrix for {self.n_estimators} trees...")
        
        for tree in self.trees:
            # Get hard labels from the current tree
            labels = tree.predict(X)
            
            # Vectorized similarity update: 
            # If labels match, they belong to the same leaf
            for cluster_id in np.unique(labels):
                mask = (labels == cluster_id)
                # Increment similarity for all pairs within the same leaf
                co_association[np.ix_(mask, mask)] += 1
        
        # Normalize similarity by number of trees
        co_association /= self.n_estimators
        
        # Final Step: Consensus via Spectral Clustering
        # This converts the similarity matrix into stable, final clusters.
        print(f"Performing Spectral Consensus for {self.n_clusters} clusters...")
        consensus_model = SpectralClustering(
            n_clusters=self.n_clusters,
            affinity='precomputed',
            random_state=self.random_state,
            assign_labels='discretize'
        )
        
        return consensus_model.fit_predict(co_association)