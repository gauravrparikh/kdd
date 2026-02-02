import numpy as np

class BaseSplitStrategy:
    """Base class for spatial split logic."""
    def propose(self, data):
        raise NotImplementedError
    
    def apply(self, data, params):
        raise NotImplementedError

class AxisSplit(BaseSplitStrategy):
    def propose(self, data):
        axis = np.random.choice(["x", "y"])
        lo, hi = data[axis].min(), data[axis].max()
        if hi - lo < 1e-5: return None
        return {"axis": axis, "value": np.random.uniform(lo, hi)}

    def apply(self, data, params):
        return data[params["axis"]] < params["value"]

class RadialSplit(BaseSplitStrategy):
    def propose(self, data):
        sample_point = data.sample(1)
        cx, cy = sample_point["x"].values[0], sample_point["y"].values[0]
        
        dists = np.sqrt((data["x"] - cx)**2 + (data["y"] - cy)**2)
        if dists.max() < 1e-5: return None
        low, high = np.percentile(dists, [10, 90])
        return {"cx": cx, "cy": cy, "r": np.random.uniform(low, high)}
    def apply(self, data, params):
        dists = np.sqrt((data["x"] - params["cx"])**2 + (data["y"] - params["cy"])**2)
        return dists < params["r"]

class ObliqueSplit(BaseSplitStrategy):
    def propose(self, data):
        # Increase variety of angles
        theta = np.random.uniform(0, np.pi) 
        a, b = np.cos(theta), np.sin(theta)
        proj = a * data["x"] + b * data["y"]
        
        if proj.max() - proj.min() < 1e-5: return None
        
        # Use quantiles to ensure the split actually divides data points
        # instead of picking a value outside the range
        low, high = np.percentile(proj, [10, 90])
        return {"a": a, "b": b, "c": np.random.uniform(low, high)}

    def apply(self, data, params):
        return (params["a"] * data["x"] + params["b"] * data["y"]) < params["c"]

class EllipticalSplit(BaseSplitStrategy):
    def propose(self, data):
        # Pick a center point from the data to ensure relevance
        sample_point = data.sample(1)
        cx, cy = sample_point["x"].values[0], sample_point["y"].values[0]
        
        # Propose rotation and axis lengths
        angle = np.random.uniform(0, np.pi)
        # major/minor axes based on local data spread
        max_dist = np.sqrt((data["x"] - cx)**2 + (data["y"] - cy)**2).max()
        if max_dist < 1e-5: return None
        
        a = np.random.uniform(0.1, max_dist)
        b = np.random.uniform(0.1, max_dist)
        
        return {"cx": cx, "cy": cy, "a": a, "b": b, "angle": angle}

    def apply(self, data, params):
        cos_a = np.cos(params["angle"])
        sin_a = np.sin(params["angle"])
        
        # Shift coordinates to center
        dx = data["x"] - params["cx"]
        dy = data["y"] - params["cy"]
        
        # Rotate coordinates
        x_rot = dx * cos_a + dy * sin_a
        y_rot = -dx * sin_a + dy * cos_a
        
        # Elliptical distance formula: (x/a)^2 + (y/b)^2 < 1
        mask = (x_rot / params["a"])**2 + (y_rot / params["b"])**2 < 1
        return mask

STRATEGY_MAP = {
    "axis": AxisSplit(),
    "radial": RadialSplit(),
    "oblique": ObliqueSplit(),
    "elliptical": EllipticalSplit()
}