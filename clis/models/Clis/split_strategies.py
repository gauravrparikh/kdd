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
        # Off-center split: use a jittered median or a wide-range uniform
        # Using 25th-75th percentile range helps avoid the exact center
        low_q, high_q = np.percentile(data[axis], [20, 80])
        return {"axis": axis, "value": np.random.uniform(low_q, high_q)}

    def apply(self, data, params):
        return data[params["axis"]] < params["value"]

class RadialSplit(BaseSplitStrategy):
    def propose(self, data):
        # To handle symmetry, pick a point and apply a "jitter" offset 
        # so the circle isn't perfectly centered on the data centroid
        sample_point = data.sample(1)
        cx, cy = sample_point["x"].values[0], sample_point["y"].values[0]
        
        # Add spatial jitter (5-10% of local spread) to break symmetry
        x_range = data["x"].max() - data["x"].min()
        y_range = data["y"].max() - data["y"].min()
        cx += np.random.uniform(-0.1, 0.1) * x_range
        cy += np.random.uniform(-0.1, 0.1) * y_range
        
        dists = np.sqrt((data["x"] - cx)**2 + (data["y"] - cy)**2)
        if dists.max() < 1e-5: return None
        
        # Use a wider range for r to allow "off-center" rings to capture edges
        low, high = np.percentile(dists, [10, 85])
        return {"cx": cx, "cy": cy, "r": np.random.uniform(low, high)}

    def apply(self, data, params):
        dists = np.sqrt((data["x"] - params["cx"])**2 + (data["y"] - params["cy"])**2)
        return dists < params["r"]

class ObliqueSplit(BaseSplitStrategy):
    def propose(self, data):
        theta = np.random.uniform(0, np.pi) 
        a, b = np.cos(theta), np.sin(theta)
        proj = a * data["x"] + b * data["y"]
        
        if proj.max() - proj.min() < 1e-5: return None
        
        # Shift the intercept 'c' away from the mean/median
        low, high = np.percentile(proj, [15, 85])
        return {"a": a, "b": b, "c": np.random.uniform(low, high)}

    def apply(self, data, params):
        return (params["a"] * data["x"] + params["b"] * data["y"]) < params["c"]

class EllipticalSplit(BaseSplitStrategy):
    def propose(self, data):
        sample_point = data.sample(1)
        cx, cy = sample_point["x"].values[0], sample_point["y"].values[0]
        
        # Off-center bias: shift center away from the sampled point slightly
        x_std = data["x"].std()
        y_std = data["y"].std()
        cx += np.random.normal(0, 0.2 * x_std)
        cy += np.random.normal(0, 0.2 * y_std)
        
        angle = np.random.uniform(0, np.pi)
        max_dist = np.sqrt((data["x"] - cx)**2 + (data["y"] - cy)**2).max()
        if max_dist < 1e-5: return None
        
        # Ensure 'a' and 'b' are not just the max distance (which creates a circle)
        # Randomizing eccentricity helps break symmetric patterns
        a = np.random.uniform(0.2, 0.9) * max_dist
        b = np.random.uniform(0.2, 0.9) * max_dist
        
        return {"cx": cx, "cy": cy, "a": a, "b": b, "angle": angle}

    def apply(self, data, params):
        cos_a = np.cos(params["angle"])
        sin_a = np.sin(params["angle"])
        dx = data["x"] - params["cx"]
        dy = data["y"] - params["cy"]
        
        x_rot = dx * cos_a + dy * sin_a
        y_rot = -dx * sin_a + dy * cos_a
        
        mask = (x_rot / params["a"])**2 + (y_rot / params["b"])**2 < 1
        return mask

STRATEGY_MAP = {
    "axis": AxisSplit(),
    "radial": RadialSplit(),
    "oblique": ObliqueSplit(),
    "elliptical": EllipticalSplit()
}