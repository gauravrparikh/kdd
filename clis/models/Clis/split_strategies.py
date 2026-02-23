"""
Generalized split strategies for arbitrary-dimensional data.

Each strategy operates on a configurable set of split columns (split_cols).
Strategies support 2D (x,y), 3D (x,y,z), or any number of dimensions.
"""

import numpy as np


class BaseSplitStrategy:
    """Base class for split logic. Operates on configurable split columns."""
    
    def __init__(self, split_cols=None):
        """
        Parameters
        ----------
        split_cols : list of str or int, optional
            Column names or indices for splitting. If None, uses ["x", "y"] for backward compat.
        """
        self.split_cols = split_cols or ["x", "y"]
    
    def _get_split_data(self, data):
        """Extract split columns as array."""
        try:
            return data[self.split_cols].values
        except (KeyError, TypeError):
            return data.iloc[:, self.split_cols].values
    
    def propose(self, data):
        raise NotImplementedError
    
    def apply(self, data, params):
        raise NotImplementedError


class AxisSplit(BaseSplitStrategy):
    """Split along a single axis (dimension)."""
    
    def propose(self, data):
        split_data = self._get_split_data(data)
        axis_idx = np.random.randint(0, split_data.shape[1])
        col_data = split_data[:, axis_idx]
        
        lo, hi = col_data.min(), col_data.max()
        if hi - lo < 1e-5:
            return None
        low_q, high_q = np.percentile(col_data, [20, 80])
        return {"axis_idx": axis_idx, "value": np.random.uniform(low_q, high_q)}

    def apply(self, data, params):
        axis_idx = params["axis_idx"]
        col = data.iloc[:, axis_idx]
        return col < params["value"]


class RadialSplit(BaseSplitStrategy):
    """Split by Euclidean distance from a center point in split space."""
    
    def propose(self, data):
        split_data = self._get_split_data(data)
        n_dims = split_data.shape[1]
        
        # Sample a point and add jitter
        sample_idx = np.random.randint(len(split_data))
        center = split_data[sample_idx].copy()
        
        ranges = split_data.max(axis=0) - split_data.min(axis=0)
        center += np.random.uniform(-0.1, 0.1, n_dims) * np.maximum(ranges, 1e-6)
        
        dists = np.sqrt(np.sum((split_data - center) ** 2, axis=1))
        if dists.max() < 1e-5:
            return None
        
        low, high = np.percentile(dists, [10, 85])
        return {"center": center, "r": np.random.uniform(low, high), "split_cols": self.split_cols}

    def apply(self, data, params):
        split_data = self._get_split_data(data)
        center = np.asarray(params["center"])
        dists = np.sqrt(np.sum((split_data - center) ** 2, axis=1))
        return dists < params["r"]


class ObliqueSplit(BaseSplitStrategy):
    """Split by linear combination (hyperplane) in split space."""
    
    def propose(self, data):
        split_data = self._get_split_data(data)
        n_dims = split_data.shape[1]
        
        # Random direction on unit sphere
        direction = np.random.randn(n_dims)
        direction /= np.linalg.norm(direction)
        
        proj = np.dot(split_data, direction)
        if proj.max() - proj.min() < 1e-5:
            return None
        
        low, high = np.percentile(proj, [15, 85])
        return {"direction": direction, "c": np.random.uniform(low, high), "split_cols": self.split_cols}

    def apply(self, data, params):
        split_data = self._get_split_data(data)
        direction = np.asarray(params["direction"])
        proj = np.dot(split_data, direction)
        return proj < params["c"]


class EllipticalSplit(BaseSplitStrategy):
    """Split by ellipse (2D) or ellipsoid (higher D) in split space."""
    
    def propose(self, data):
        split_data = self._get_split_data(data)
        n_dims = split_data.shape[1]
        
        if n_dims < 2:
            return None
            
        # Sample center with jitter
        sample_idx = np.random.randint(len(split_data))
        center = split_data[sample_idx].copy()
        stds = np.std(split_data, axis=0)
        center += np.random.normal(0, 0.2, n_dims) * np.maximum(stds, 1e-6)
        
        max_dist = np.sqrt(np.sum((split_data - center) ** 2, axis=1)).max()
        if max_dist < 1e-5:
            return None
        
        # Semi-axes (random eccentricity)
        axes = np.random.uniform(0.2, 0.9, n_dims) * max_dist
        
        # Random rotation for n_dims >= 2
        if n_dims == 2:
            angle = np.random.uniform(0, np.pi)
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            rotation = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        else:
            # Random orthogonal matrix (simplified: use QR of random matrix)
            rotation = np.linalg.qr(np.random.randn(n_dims, n_dims))[0]
        
        return {
            "center": center,
            "axes": axes,
            "rotation": rotation,
            "split_cols": self.split_cols,
            "n_dims": n_dims
        }

    def apply(self, data, params):
        split_data = self._get_split_data(data)
        center = np.asarray(params["center"])
        axes = np.asarray(params["axes"])
        rotation = np.asarray(params["rotation"])
        
        centered = split_data - center
        rotated = np.dot(centered, rotation.T)
        normalized = rotated / np.maximum(axes, 1e-10)
        return np.sum(normalized ** 2, axis=1) < 1


def get_strategy_map(split_cols=None):
    """
    Get strategy instances configured for the given split columns.
    
    Parameters
    ----------
    split_cols : list, optional
        Column names/indices for splitting. Default ["x", "y"].
    
    Returns
    -------
    dict : strategy_name -> strategy instance
    """
    cols = split_cols or ["x", "y"]
    return {
        "axis": AxisSplit(cols),
        "radial": RadialSplit(cols),
        "oblique": ObliqueSplit(cols),
        "elliptical": EllipticalSplit(cols),
    }


# Backward compatibility: default 2D strategies
STRATEGY_MAP = get_strategy_map(["x", "y"])
