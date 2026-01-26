import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClusterMixin
from .split_strategies import STRATEGY_MAP

class Clis(BaseEstimator, ClusterMixin):
    def __init__(
        self,
        min_samples_leaf=30,
        gain_threshold=0.01,
        loss_metric="nll",
        strategies=("axis", "radial", "oblique"),
        random_state=42,
    ):
        self.min_samples_leaf = min_samples_leaf
        self.gain_threshold = gain_threshold
        self.loss_metric = loss_metric
        self.strategies = strategies
        self.random_state = random_state
        
        self.tree_ = {}
        self.leaf_labels_ = {}
        self._next_node_id = 0

    def _calculate_loss(self, y):
        n = len(y)
        if n < 2: return 0.0
        
        if self.loss_metric == "mse":
            return np.sum((y - np.mean(y))**2)
        
        elif self.loss_metric == "nll":
            var = np.var(y)
            # Threshold variance to avoid log(0)
            return (n / 2) * np.log(max(var, 1e-6)) + (n / 2)
        
        elif self.loss_metric == "pinball":
            loss = 0.0
            for q in [0.1, 0.5, 0.9]:
                pred = np.percentile(y, q * 100)
                resid = y - pred
                loss += np.sum(np.maximum(q * resid, (q - 1) * resid))
            return loss
        return 0.0

    def _find_best_split(self, data, indices):
        sub_data = data.iloc[indices]
        parent_loss = self._calculate_loss(sub_data["z"].values)
        
        best_gain = -np.inf
        best_split_info = None
        best_children = None
        
        # Hyperparameter: Number of random split attempts per node
        n_attempts = 40
        
        for _ in range(n_attempts):
            s_name = np.random.choice(self.strategies)
            strategy = STRATEGY_MAP[s_name]
            
            params = strategy.propose(sub_data)
            if params is None: continue
            
            mask = strategy.apply(sub_data, params)
            left_idx = np.array(indices)[mask]
            right_idx = np.array(indices)[~mask]
            
            if len(left_idx) < self.min_samples_leaf or len(right_idx) < self.min_samples_leaf:
                continue
            
            loss_l = self._calculate_loss(data.iloc[left_idx]["z"].values)
            loss_r = self._calculate_loss(data.iloc[right_idx]["z"].values)
            gain = parent_loss - (loss_l + loss_r)
            
            if gain > best_gain:
                best_gain = gain
                best_split_info = (s_name, params)
                best_children = (left_idx.tolist(), right_idx.tolist())
                
        return best_split_info, best_gain, best_children

    def fit(self, X, y=None):
        """
        X should be a DataFrame with 'x' and 'y' columns.
        'z' (target variance) must be included in X for this specific unsupervised task.
        """
        np.random.seed(self.random_state)
        self.tree_ = {}
        self.leaf_labels_ = {}
        self._next_node_id = 1
        
        initial_loss = self._calculate_loss(X["z"].values)
        eff_threshold = self.gain_threshold * abs(initial_loss) if initial_loss != 0 else 0.001
        
        # Queue for Breadth-First Tree Construction: (node_id, indices)
        queue = [(0, list(range(len(X))))]
        
        while queue:
            node_id, indices = queue.pop(0)
            
            if len(indices) >= 2 * self.min_samples_leaf:
                split_info, gain, children = self._find_best_split(X, indices)
                
                if split_info is not None and gain >= eff_threshold:
                    left_id = self._next_node_id
                    right_id = self._next_node_id + 1
                    self._next_node_id += 2
                    
                    self.tree_[node_id] = (split_info, left_id, right_id)
                    queue.append((left_id, children[0]))
                    queue.append((right_id, children[1]))
                    continue
            
            # If no split, mark as leaf
            self.leaf_labels_[node_id] = node_id
            
        return self

    def predict(self, X):
        """Traverse the tree to assign labels to new spatial points."""
        labels = np.zeros(len(X), dtype=int)
        
        for i in range(len(X)):
            row = X.iloc[i]
            node = 0
            while node in self.tree_:
                (s_name, params), left, right = self.tree_[node]
                strategy = STRATEGY_MAP[s_name]
                
                # We wrap the row in a DataFrame-like structure for the strategy
                go_left = strategy.apply(pd.DataFrame([row]), params).iloc[0]
                node = left if go_left else right
            
            labels[i] = self.leaf_labels_.get(node, -1)
        return labels