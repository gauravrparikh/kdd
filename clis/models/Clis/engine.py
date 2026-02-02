import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClusterMixin
from .split_strategies import STRATEGY_MAP

class Clis(BaseEstimator, ClusterMixin):
    def __init__(
        self,
        min_samples_leaf=10,
        gain_threshold=0.05,
        loss_metric="pinball", 
        strategies=("axis", "radial", "oblique", "elliptical"),
        random_state=42,
        complexity_penalty=2.0,
        lookahead_depth=2
    ):
        self.complexity_penalty = complexity_penalty
        self.min_samples_leaf = min_samples_leaf
        self.gain_threshold = gain_threshold
        self.loss_metric = loss_metric
        self.strategies = strategies
        self.random_state = random_state
        self.lookahead_depth = lookahead_depth
        
        self.tree_ = {}
        self.leaf_labels_ = {}
        self._next_node_id = 0

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
            for q in [0.1, 0.5, 0.9]:
                pred = np.percentile(y, q * 100)
                resid = y - pred
                loss += np.sum(np.maximum(q * resid, (q - 1) * resid))
            return loss
        return 0.0

    def _evaluate_lookahead(self, data, indices, current_lookahead):
        """
        Recursively evaluates the best possible gain from the current node 
        up to the lookahead_depth.
        """
        sub_data = data.iloc[indices]
        n_node = len(indices)
        parent_loss = self._calculate_loss(sub_data["z"].values)
        
        best_path_gain = -np.inf
        best_split_info = None
        best_children = None
        
        # Base penalty for making any split
        split_penalty = self.complexity_penalty * np.log(n_node)
        
        for strategy_name in self.strategies:
            strategy = STRATEGY_MAP[strategy_name]
            # Test 10 proposals per strategy to find a path
            for _ in range(10):
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
                
                # Look-ahead: If immediate gain is low, check deeper
                path_gain = immediate_gain
                if current_lookahead < self.lookahead_depth:
                    print("Evaluating lookahead at depth", current_lookahead + 1)
                    # Recursive call to see if children have hidden gain
                    _, left_gain, _ = self._evaluate_lookahead(data, left_idx.tolist(), current_lookahead + 1)
                    _, right_gain, _ = self._evaluate_lookahead(data, right_idx.tolist(), current_lookahead + 1)
                    
                    # Add the potential of the best future sub-splits
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
        self._next_node_id = 1
        
        initial_loss = self._calculate_loss(data_internal["z"].values)
        eff_threshold = self.gain_threshold * abs(initial_loss) if initial_loss != 0 else 0.001
        
        # Queue for standard Breadth-First Growth
        queue = [(0, list(range(len(data_internal))))]
        
        while queue:
            node_id, indices = queue.pop(0)
            
            if len(indices) >= 2 * self.min_samples_leaf:
                # Use lookahead-aware evaluator (starting at lookahead depth 0)
                split_info, path_gain, children = self._evaluate_lookahead(data_internal, indices, 0)
                
                if split_info is not None and path_gain >= eff_threshold:
                    left_id = self._next_node_id
                    right_id = self._next_node_id + 1
                    self._next_node_id += 2
                    
                    self.tree_[node_id] = (split_info, left_id, right_id)
                    queue.append((left_id, children[0]))
                    queue.append((right_id, children[1]))
                    continue
            
            self.leaf_labels_[node_id] = node_id
            
        return self

    def predict(self, X):
        """
        Traverse the tree to assign labels to new spatial points using vectorization.
        Instead of row-by-row iteration, we process all points simultaneously 
        at each node using boolean masks.
        """
        n_samples = len(X)
        labels = np.zeros(n_samples, dtype=int)
        
        # We use a queue to track groups of indices that belong to specific nodes
        # Queue item: (node_id, array_of_indices_at_this_node)
        queue = [(0, np.arange(n_samples))]
        
        while queue:
            node_id, current_indices = queue.pop(0)
            
            if len(current_indices) == 0:
                continue
                
            # If current node is in the tree, it has children (it's a split node)
            if node_id in self.tree_:
                (s_name, params), left_id, right_id = self.tree_[node_id]
                strategy = STRATEGY_MAP[s_name]
                
                # Apply strategy to the subset of data at this node
                # strategy.apply returns a boolean mask for the provided data
                mask = strategy.apply(X.iloc[current_indices], params).values
                
                # Split current indices based on the mask
                left_indices = current_indices[mask]
                right_indices = current_indices[~mask]
                
                queue.append((left_id, left_indices))
                queue.append((right_id, right_indices))
            
            # If it's a leaf node, assign the leaf label to all indices here
            elif node_id in self.leaf_labels_:
                labels[current_indices] = self.leaf_labels_[node_id]
            
            else:
                # Fallback for unexpected node paths
                labels[current_indices] = -1
                
        return labels