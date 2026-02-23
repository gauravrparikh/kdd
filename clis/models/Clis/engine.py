"""
CLIS: Clustering with Loss-based Independence Splitting.

Generalized engine supporting:
- Arbitrary split dimensions (split_cols)
- Single or joint target distributions (y can be 1D or 2D)
- MMD for merging (joint distributions) or KS (1D only)
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClusterMixin
from scipy.stats import ks_2samp

from .split_strategies import get_strategy_map
from .metrics.mmd import mmd_squared, mmd_null_pvalue


class Clis(BaseEstimator, ClusterMixin):
    def __init__(
        self,
        split_cols=None,
        min_samples_leaf=10,
        gain_threshold=0.001,
        loss_metric="pinball",
        strategies=("axis", "radial", "oblique", "elliptical"),
        random_state=42,
        complexity_penalty=1.0,
        lookahead_depth=2,
        merge_threshold=0.005,
        merge_metric="auto",
        min_depth=1,
        mmd_n_permutations=25,
        merge_use_permutation=False,
        merge_mmd_threshold=0.1,
    ):
        """
        Parameters
        ----------
        split_cols : list of str or int, optional
            Columns of X used for splitting. If None, uses all columns of X.
        min_samples_leaf : int
            Minimum samples per leaf.
        gain_threshold : float
            Minimum gain to accept a split.
        loss_metric : str
            "mse", "nll", or "pinball" for target loss.
        strategies : tuple
            Split strategy names to use.
        merge_threshold : float
            For KS: merge when p-value > threshold. For MMD: merge when p-value > threshold.
        merge_metric : str
            "ks" (1D only), "mmd" (any dim), or "auto" (mmd if y is 2D, else ks).
        mmd_n_permutations : int
            Permutations for MMD p-value when merge_metric="mmd" and merge_use_permutation=True.
        merge_use_permutation : bool
            If False (faster), merge when MMD^2 < merge_mmd_threshold. If True, use permutation p-value.
        merge_mmd_threshold : float
            When merge_use_permutation=False, merge leaves when MMD^2 < this value.
        """
        self.split_cols = split_cols
        self.complexity_penalty = complexity_penalty
        self.min_samples_leaf = min_samples_leaf
        self.gain_threshold = gain_threshold
        self.loss_metric = loss_metric
        self.strategies = strategies
        self.random_state = random_state
        self.lookahead_depth = lookahead_depth
        self.merge_threshold = merge_threshold
        self.merge_metric = merge_metric
        self.mmd_n_permutations = mmd_n_permutations
        self.merge_use_permutation = merge_use_permutation
        self.merge_mmd_threshold = merge_mmd_threshold

        self.tree_ = {}
        self.leaf_labels_ = {}
        self.merge_map_ = {}
        self._next_node_id = 0
        self.min_depth = min_depth

    def _resolve_split_cols(self, X):
        """Resolve split_cols from X."""
        if self.split_cols is not None:
            return list(self.split_cols)
        if isinstance(X, pd.DataFrame):
            return list(X.columns)
        return list(range(X.shape[1]))

    def _ensure_dataframe(self, X, split_cols):
        """Ensure X is DataFrame with proper columns."""
        if isinstance(X, pd.DataFrame):
            return X
        n_cols = X.shape[1]
        cols = split_cols if len(split_cols) == n_cols else [f"d{i}" for i in range(n_cols)]
        return pd.DataFrame(X, columns=cols)

    def score(self, X, y):
        """Internal scorer for GridSearchCV: Lower NLL is better."""
        labels = self.predict(X)
        total_nll = 0
        y_arr = np.atleast_2d(y).T if np.ndim(y) == 1 else np.asarray(y)
        for lab in np.unique(labels):
            if lab == -1:
                continue
            z_cluster = y_arr[labels == lab]
            if len(z_cluster) > 1:
                if z_cluster.shape[1] == 1:
                    var = np.var(z_cluster)
                else:
                    var = np.linalg.det(np.cov(z_cluster.T)) or 1e-6
                total_nll += (len(z_cluster) / 2) * np.log(max(var, 1e-6))
        return -total_nll

    def _calculate_loss(self, y):
        """
        Loss for a group. Supports 1D (single variable) and multi-D (joint distribution).

        For 1D: uses loss_metric (mse, nll, pinball).
        For multi-D: always uses multivariate Gaussian NLL — pinball/mse do not
        capture joint distributions (correlations, covariance structure).
        """
        y = np.asarray(y)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        n, d = y.shape
        if n < self.min_samples_leaf:
            return 0.0

        # Multidimensional outcome: use multivariate NLL only (captures joint distribution)
        if d > 1:
            cov = np.cov(y.T)
            if cov.ndim == 0:
                cov = np.array([[cov]])
            det = np.linalg.det(cov)
            if det <= 0 or not np.isfinite(det):
                det = 1e-6
            return (n / 2) * np.log(det) + (n * d / 2)

        # 1D outcome: use configured loss_metric
        if self.loss_metric == "mse":
            return np.sum((y - np.mean(y, axis=0)) ** 2)
        elif self.loss_metric == "nll":
            var = np.var(y)
            return (n / 2) * np.log(max(var, 1e-6)) + (n / 2)
        elif self.loss_metric == "pinball":
            loss = 0.0
            for q in [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]:
                pred = np.percentile(y[:, 0], q * 100)
                resid = y[:, 0] - pred
                loss += np.sum(np.maximum(q * resid, (q - 1) * resid))
            return loss
        return 0.0

    def _evaluate_lookahead(self, data, indices, split_cols, target_cols, strategy_map, current_lookahead=0, current_depth=0):
        """Evaluates the best gain from the current node."""
        sub_data = data.iloc[indices]
        n_node = len(indices)
        parent_loss = self._calculate_loss(sub_data[target_cols].values)

        best_path_gain = -np.inf
        best_split_info = None
        best_children = None

        split_penalty = self.complexity_penalty * np.log(n_node)
        n_proposals = 20 if current_depth == 0 else 5

        for strategy_name in self.strategies:
            if strategy_name not in strategy_map:
                continue
            strategy = strategy_map[strategy_name]
            for _ in range(n_proposals):
                params = strategy.propose(sub_data)
                if params is None:
                    continue

                mask = strategy.apply(sub_data, params)
                mask = mask.values if hasattr(mask, 'values') else np.asarray(mask)
                left_idx = np.array(indices)[mask]
                right_idx = np.array(indices)[~mask]

                if len(left_idx) < self.min_samples_leaf or len(right_idx) < self.min_samples_leaf:
                    continue

                loss_l = self._calculate_loss(data.iloc[left_idx][target_cols].values)
                loss_r = self._calculate_loss(data.iloc[right_idx][target_cols].values)
                immediate_gain = parent_loss - (loss_l + loss_r) - split_penalty

                if hasattr(self, "min_depth") and current_depth < self.min_depth:
                    path_gain = max(immediate_gain, 1e-5)
                else:
                    path_gain = immediate_gain

                clear_winner_threshold = self.gain_threshold * 5

                if path_gain > clear_winner_threshold:
                    pass
                elif current_lookahead < self.lookahead_depth:
                    _, left_gain, _ = self._evaluate_lookahead(
                        data, left_idx.tolist(), split_cols, target_cols, strategy_map,
                        current_lookahead + 1, current_depth + 1
                    )
                    _, right_gain, _ = self._evaluate_lookahead(
                        data, right_idx.tolist(), split_cols, target_cols, strategy_map,
                        current_lookahead + 1, current_depth + 1
                    )
                    path_gain += max(0, left_gain) + max(0, right_gain)

                if path_gain > best_path_gain:
                    best_path_gain = path_gain
                    best_split_info = (strategy_name, params)
                    best_children = (left_idx.tolist(), right_idx.tolist())

        return best_split_info, best_path_gain, best_children

    def fit(self, X, y):
        # Resolve split columns and ensure DataFrame
        split_cols = self._resolve_split_cols(X)
        X = self._ensure_dataframe(X, split_cols)

        # Build internal data: split cols + target
        y = np.asarray(y)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        n_targets = y.shape[1]
        target_cols = [f"_z{i}" for i in range(n_targets)]
        data_internal = X[split_cols].copy()
        for i, tc in enumerate(target_cols):
            data_internal[tc] = y[:, i]

        # Merge metric: auto = mmd for multi-target, ks for single
        merge_metric = self.merge_metric
        if merge_metric == "auto":
            merge_metric = "mmd" if n_targets > 1 else "ks"

        self._split_cols = split_cols
        self._target_cols = target_cols
        self._merge_metric = merge_metric
        self._strategy_map = get_strategy_map(split_cols)

        np.random.seed(self.random_state)
        self.tree_ = {}
        self.leaf_labels_ = {}
        self.merge_map_ = {}
        self._next_node_id = 1

        initial_loss = self._calculate_loss(data_internal[target_cols].values)
        eff_threshold = self.gain_threshold * abs(initial_loss) if initial_loss != 0 else 0.001

        queue = [(0, list(range(len(data_internal))))]
        leaf_data_map = {}

        # Part 1: Recursive Splitting
        while queue:
            node_id, indices = queue.pop(0)
            if len(indices) >= 2 * self.min_samples_leaf:
                split_info, path_gain, children = self._evaluate_lookahead(
                    data_internal, indices, split_cols, target_cols, self._strategy_map, 0, 0
                )

                if split_info is not None and path_gain >= eff_threshold:
                    left_id, right_id = self._next_node_id, self._next_node_id + 1
                    self._next_node_id += 2
                    self.tree_[node_id] = (split_info, left_id, right_id)
                    queue.append((left_id, children[0]))
                    queue.append((right_id, children[1]))
                    continue

            self.leaf_labels_[node_id] = node_id
            leaf_data_map[node_id] = data_internal.iloc[indices][target_cols].values

        # Part 2: Merging statistically similar leaves
        self._perform_merging(leaf_data_map, merge_metric)
        return self

    def _perform_merging(self, leaf_data_map, merge_metric):
        leaf_ids = list(leaf_data_map.keys())
        parent = {lid: lid for lid in leaf_ids}

        def find(i):
            if parent[i] == i:
                return i
            parent[i] = find(parent[i])
            return parent[i]

        def union(i, j):
            root_i, root_j = find(i), find(j)
            if root_i != root_j:
                parent[root_i] = root_j

        for i in range(len(leaf_ids)):
            for j in range(i + 1, len(leaf_ids)):
                id_a, id_b = leaf_ids[i], leaf_ids[j]
                data_a = leaf_data_map[id_a]
                data_b = leaf_data_map[id_b]

                if merge_metric == "ks":
                    if data_a.ndim > 1 and data_a.shape[1] > 1:
                        p_val = 0.0
                    else:
                        a_1d = data_a.flatten()
                        b_1d = data_b.flatten()
                        _, p_val = ks_2samp(a_1d, b_1d)
                    should_merge = p_val > self.merge_threshold
                else:
                    # MMD: use threshold (fast) or permutation p-value (slower)
                    if self.merge_use_permutation:
                        _, p_val = mmd_null_pvalue(
                            data_a, data_b,
                            n_permutations=self.mmd_n_permutations,
                            random_state=self.random_state,
                        )
                        should_merge = p_val > self.merge_threshold
                    else:
                        mmd2 = mmd_squared(data_a, data_b)
                        should_merge = mmd2 < self.merge_mmd_threshold

                if should_merge:
                    union(id_a, id_b)

        self.merge_map_ = {lid: find(lid) for lid in leaf_ids}

    def predict(self, X):
        X = self._ensure_dataframe(X, self._split_cols)
        n_samples = len(X)
        labels = np.zeros(n_samples, dtype=int)
        queue = [(0, np.arange(n_samples))]

        while queue:
            node_id, current_indices = queue.pop(0)
            if len(current_indices) == 0:
                continue

            if node_id in self.tree_:
                (s_name, params), left_id, right_id = self.tree_[node_id]
                strategy = self._strategy_map[s_name]
                mask = strategy.apply(X.iloc[current_indices], params)
                mask = mask.values if hasattr(mask, 'values') else np.asarray(mask)
                queue.append((left_id, current_indices[mask]))
                queue.append((right_id, current_indices[~mask]))
            elif node_id in self.leaf_labels_:
                labels[current_indices] = self.merge_map_.get(node_id, node_id)
            else:
                labels[current_indices] = -1
        return labels
