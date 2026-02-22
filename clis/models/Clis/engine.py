import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClusterMixin
from scipy.stats import ks_2samp
from .split_strategies import STRATEGY_MAP

class Clis(BaseEstimator, ClusterMixin):
    def __init__(
        self,
        min_samples_leaf=10,
        gain_threshold=0.001,
        loss_metric="pinball", 
        strategies=("axis", "radial", "oblique", "elliptical"),
        random_state=42,
        complexity_penalty=1.0,
        lookahead_depth=2,
        merge_threshold=0.005,
        min_depth=1
    ):
        self.complexity_penalty = complexity_penalty
        self.min_samples_leaf = min_samples_leaf
        self.gain_threshold = gain_threshold
        self.loss_metric = loss_metric
        self.strategies = strategies
        self.random_state = random_state
        self.lookahead_depth = lookahead_depth
        self.merge_threshold = merge_threshold
        
        self.tree_ = {}
        self.leaf_labels_ = {}
        self.merge_map_ = {}
        self._next_node_id = 0
        self.min_depth= min_depth
        
    def score(self, X, y):
        """Internal scorer for GridSearchCV: Lower NLL is better."""
        labels = self.predict(X)
        total_nll = 0
        for lab in np.unique(labels):
            z_cluster = y[labels == lab]
            if len(z_cluster) > 1:
                var = np.var(z_cluster)
                total_nll += (len(z_cluster) / 2) * np.log(max(var, 1e-6))
        return -total_nll # Negative because GridSearchCV maximizes
    
    def _calculate_loss(self, y):
        n = len(y)
        if n < self.min_samples_leaf: return 0.0
        
        if self.loss_metric == "mse":
            return np.sum((y - np.mean(y))**2)
        elif self.loss_metric == "nll":
            var = np.var(y)
            return (n / 2) * np.log(max(var, 1e-6)) + (n / 2)
        elif self.loss_metric == "pinball":
            loss = 0.0
            for q in [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]:
                pred = np.percentile(y, q * 100)
                resid = y - pred
                loss += np.sum(np.maximum(q * resid, (q - 1) * resid))
            return loss
        return 0.0


    def _evaluate_lookahead(self, data, indices, current_lookahead, current_depth=0):
        """
        Evaluates the best gain from the current node with optimized computational efficiency.
        """
        sub_data = data.iloc[indices]
        n_node = len(indices)
        parent_loss = self._calculate_loss(sub_data["z"].values)
        
        best_path_gain = -np.inf
        best_split_info = None
        best_children = None
        
        # Base penalty for making any split
        split_penalty = self.complexity_penalty * np.log(n_node)
        
        # Adaptive Sampling: Use 50 proposals at the root for accuracy, 
        # and 10 at deeper levels to save computation.
        n_proposals = 50 if current_depth == 0 else 10
        
        for strategy_name in self.strategies:
            strategy = STRATEGY_MAP[strategy_name]
            for _ in range(n_proposals):
                params = strategy.propose(sub_data)
                if params is None: continue
                
                mask = strategy.apply(sub_data, params).values
                left_idx = np.array(indices)[mask]
                right_idx = np.array(indices)[~mask]
                
                if len(left_idx) < self.min_samples_leaf or len(right_idx) < self.min_samples_leaf:
                    continue
                
                # Immediate Gain
                loss_l = self._calculate_loss(data.iloc[left_idx]["z"].values)
                loss_r = self._calculate_loss(data.iloc[right_idx]["z"].values)
                immediate_gain = parent_loss - (loss_l + loss_r) - split_penalty
                
                # Warm Start: Force a positive signal if below min_depth to prevent early stopping.
                if hasattr(self, 'min_depth') and current_depth < self.min_depth:
                    path_gain = max(immediate_gain, 1e-5)
                else:
                    path_gain = immediate_gain

                # Early Exit: Skip expensive lookahead if the immediate gain is clearly superior.
                clear_winner_threshold = self.gain_threshold * 5
                
                if path_gain > clear_winner_threshold:
                    # Maintain path_gain as immediate_gain
                    pass
                elif current_lookahead < self.lookahead_depth:
                    # Recursive lookahead only for marginal splits to reduce burden.
                    _, left_gain, _ = self._evaluate_lookahead(data, left_idx.tolist(), current_lookahead + 1, current_depth + 1)
                    _, right_gain, _ = self._evaluate_lookahead(data, right_idx.tolist(), current_lookahead + 1, current_depth + 1)
                    
                    # Add potential of best future sub-splits
                    path_gain += max(0, left_gain) + max(0, right_gain)
                
                if path_gain > best_path_gain:
                    best_path_gain = path_gain
                    best_split_info = (strategy_name, params)
                    best_children = (left_idx.tolist(), right_idx.tolist())
                    
        return best_split_info, best_path_gain, best_children
    
    def fit(self, X, y):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=['x', 'y'])
        
        data_internal = X.copy()
        data_internal["z"] = y 
        
        np.random.seed(self.random_state)
        self.tree_ = {}
        self.leaf_labels_ = {}
        self.merge_map_ = {}
        self._next_node_id = 1
        
        initial_loss = self._calculate_loss(data_internal["z"].values)
        eff_threshold = self.gain_threshold * abs(initial_loss) if initial_loss != 0 else 0.001
        
        queue = [(0, list(range(len(data_internal))))]
        leaf_data_map = {}

        # Part 1: Recursive Splitting
        while queue:
            node_id, indices = queue.pop(0)
            if len(indices) >= 2 * self.min_samples_leaf:
                split_info, path_gain, children = self._evaluate_lookahead(data_internal, indices, 0)
                
                if split_info is not None and path_gain >= eff_threshold:
                    left_id, right_id = self._next_node_id, self._next_node_id + 1
                    self._next_node_id += 2
                    self.tree_[node_id] = (split_info, left_id, right_id)
                    queue.append((left_id, children[0]))
                    queue.append((right_id, children[1]))
                    continue
            
            self.leaf_labels_[node_id] = node_id
            leaf_data_map[node_id] = data_internal.iloc[indices]["z"].values

        # Part 2: Merging statistically similar leaves
        self._perform_merging(leaf_data_map)
        return self

    def _perform_merging(self, leaf_data_map):
        leaf_ids = list(leaf_data_map.keys())
        parent = {lid: lid for lid in leaf_ids}

        def find(i):
            if parent[i] == i: return i
            parent[i] = find(parent[i])
            return parent[i]

        def union(i, j):
            root_i, root_j = find(i), find(j)
            if root_i != root_j: parent[root_i] = root_j

        for i in range(len(leaf_ids)):
            for j in range(i + 1, len(leaf_ids)):
                id_a, id_b = leaf_ids[i], leaf_ids[j]
                _, p_val = ks_2samp(leaf_data_map[id_a], leaf_data_map[id_b])
                
                # High p-value means we cannot distinguish the distributions
                if p_val > self.merge_threshold:
                    union(id_a, id_b)

        self.merge_map_ = {lid: find(lid) for lid in leaf_ids}

    def predict(self, X):
        n_samples = len(X)
        labels = np.zeros(n_samples, dtype=int)
        queue = [(0, np.arange(n_samples))]
        
        while queue:
            node_id, current_indices = queue.pop(0)
            if len(current_indices) == 0: continue
            
            if node_id in self.tree_:
                (s_name, params), left_id, right_id = self.tree_[node_id]
                mask = STRATEGY_MAP[s_name].apply(X.iloc[current_indices], params).values
                queue.append((left_id, current_indices[mask]))
                queue.append((right_id, current_indices[~mask]))
            elif node_id in self.leaf_labels_:
                # Map original leaf ID to merged cluster ID
                labels[current_indices] = self.merge_map_.get(node_id, node_id)
            else:
                labels[current_indices] = -1
        return labels