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
        cx = np.random.uniform(data["x"].min(), data["x"].max())
        cy = np.random.uniform(data["y"].min(), data["y"].max())
        dists = np.sqrt((data["x"] - cx)**2 + (data["y"] - cy)**2)
        if dists.max() < 1e-5: return None
        return {"cx": cx, "cy": cy, "r": np.random.uniform(0, dists.max())}

    def apply(self, data, params):
        dists = np.sqrt((data["x"] - params["cx"])**2 + (data["y"] - params["cy"])**2)
        return dists < params["r"]

class ObliqueSplit(BaseSplitStrategy):
    def propose(self, data):
        theta = np.random.uniform(0, 2 * np.pi)
        a, b = np.cos(theta), np.sin(theta)
        proj = a * data["x"] + b * data["y"]
        if proj.max() - proj.min() < 1e-5: return None
        return {"a": a, "b": b, "c": np.random.uniform(proj.min(), proj.max())}

    def apply(self, data, params):
        return (params["a"] * data["x"] + params["b"] * data["y"]) < params["c"]

# Map names to classes for easy configuration
STRATEGY_MAP = {
    "axis": AxisSplit(),
    "radial": RadialSplit(),
    "oblique": ObliqueSplit()
}