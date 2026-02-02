import numpy as np
import os
from scipy.spatial import cKDTree

class SyntheticFactory:
    def __init__(self, n_samples=50000, seed=42, mean_shift=False):
        """
        Initialize the factory.
        :param mean_shift: If True, clusters will have different means in addition to variances.
        """
        self.n_samples = n_samples
        self.seed = seed
        self.mean_shift = mean_shift
        np.random.seed(seed)
        # Pre-generate coordinates
        self.x = np.random.uniform(-15, 15, n_samples)
        self.y = np.random.uniform(-15, 15, n_samples)
        self.coords = np.column_stack((self.x, self.y))

    def _get_params(self, sigmas, means=None):
        """
        Helper to return either a global mean or specific cluster means.
        Default global mean is 50.0 to match original standard.
        """
        if not self.mean_shift or means is None:
            return sigmas, [50.0] * len(sigmas)
        return sigmas, means

    def _save(self, name, z, labels):
        # Ensure data directory exists
        os.makedirs("data", exist_ok=True)
        output_path = os.path.join("data", f"{name}.npz")
        np.savez(output_path, x=self.x, y=self.y, z=z, labels=labels)
        print(f"Saved: {name} (Mean Shift: {self.mean_shift})")

    def generate_voronoi_sharp(self):
        """Sharp boundaries based on proximity to seeds."""
        seeds = np.array([[-8, -8], [5, -5], [0, 8], [10, 10], [-10, 5]])
        tree = cKDTree(seeds)
        _, labels = tree.query(self.coords)
        
        # Define Sigmas and optional stark Means
        sigmas, means = self._get_params(
            sigmas=[0.1, 5.0, 50.0, 150.0, 300.0],
            means=[0, 100, -100, 500, -500]
        )
        
        z = np.array([np.random.normal(means[l % 5], sigmas[l % 5]) for l in labels])
        self._save("voronoi_sharp", z, labels)

    def generate_variance_gradient(self):
        """Variance increases linearly from left to right."""
        labels = (self.x - self.x.min()) / (self.x.max() - self.x.min()) * 5
        labels = labels.astype(int)
        
        # Sigma is a function of X; Mean remains 50.0 as it's a continuous gradient
        sigmas = 10 + (self.x - self.x.min()) * 15 
        z = np.random.normal(50.0, sigmas)
        self._save("linear_gradient", z, labels)

    def generate_concentric_donuts(self):
        """Variance changes based on distance from origin."""
        dist = np.sqrt(self.x**2 + self.y**2)
        labels = np.zeros(self.n_samples)
        labels[(dist > 5) & (dist <= 10)] = 1
        labels[dist > 10] = 2
        
        sigmas, means = self._get_params(
            sigmas=[2.0, 200.0, 10.0],
            means=[0, 600, -200]
        )
        
        z = np.array([np.random.normal(means[int(l)], sigmas[int(l)]) for l in labels])
        self._save("concentric_donuts", z, labels)

    def generate_anisotropic_streaks(self):
        """Variance changes along an oblique stripe pattern."""
        projection = (self.x + self.y) / np.sqrt(2)
        labels = (projection // 5).astype(int) % 2 
        
        sigmas, means = self._get_params(
            sigmas=[1.0, 100.0],
            means=[0, 400]
        )
        
        z = np.array([np.random.normal(means[l], sigmas[l]) for l in labels])
        self._save("oblique_stripes", z, labels)

    def generate_sparse_islands(self):
        """Small hotspots of high variance."""
        labels = np.zeros(self.n_samples)
        hotspots = np.array([[-10, 10], [12, -2], [0, 0]])
        for hs in hotspots:
            dist = np.sqrt((self.x - hs[0])**2 + (self.y - hs[1])**2)
            labels[dist < 3] = 1 
            
        sigmas, means = self._get_params(
            sigmas=[5.0, 400.0],
            means=[50, -300]
        )
        
        z = np.array([np.random.normal(means[int(l)], sigmas[int(l)]) for l in labels])
        self._save("sparse_islands", z, labels)

    def generate_spiral_volatility(self):
        """Variance follows an Archimedean spiral."""
        theta = np.sqrt(np.random.rand(self.n_samples)) * 4 * np.pi 
        r = theta * 1.2
        spiral_x = r * np.cos(theta)
        spiral_y = r * np.sin(theta)
        
        tree = cKDTree(np.column_stack((spiral_x, spiral_y)))
        dist, _ = tree.query(self.coords)
        
        labels = (dist < 2.5).astype(int)
        sigmas, means = self._get_params(
            sigmas=[2.0, 250.0],
            means=[0, 500]
        )
        
        z = np.array([np.random.normal(means[l], sigmas[l]) for l in labels])
        self._save("spiral_volatility", z, labels)

    def generate_checkerboard_regimes(self):
        """Classic checkerboard variance patterns."""
        grid_size = 6.0 
        x_bins = ((self.x + 15) // grid_size).astype(int)
        y_bins = ((self.y + 15) // grid_size).astype(int)
        
        labels = (x_bins + y_bins) % 2
        sigmas, means = self._get_params(
            sigmas=[0.5, 120.0],
            means=[0, 250]
        )
        
        z = np.array([np.random.normal(means[l], sigmas[l]) for l in labels])
        self._save("checkerboard", z, labels)

    def generate_density_variance_bias(self):
        """High variance in sparse regions, Low variance in dense regions."""
        core_mask = (np.sqrt(self.x**2 + self.y**2) < 7)
        labels = core_mask.astype(int)
        
        sigmas, means = self._get_params(
            sigmas=[300.0, 2.0],
            means=[400, 0]
        )
        
        z = np.array([np.random.normal(means[l], sigmas[l]) for l in labels])
        self._save("density_bias", z, labels)

    def generate_nested_clusters(self):
        """A variance cluster inside a cluster inside a sea."""
        dist = np.sqrt(self.x**2 + self.y**2)
        labels = np.zeros(self.n_samples)
        labels[dist < 10] = 1 
        labels[dist < 4] = 2  
        
        sigmas, means = self._get_params(
            sigmas=[200.0, 5.0, 200.0],
            means=[500, 0, -500]
        )
        
        z = np.array([np.random.normal(means[int(l)], sigmas[int(l)]) for l in labels])
        self._save("nested_targets", z, labels)

    def generate_fractal_noise(self):
        """Clouds of variance created by wave interference."""
        val = np.sin(self.x/3) * np.cos(self.y/3) + np.sin((self.x+self.y)/5)
        labels = (val > 0).astype(int)
        
        sigmas, means = self._get_params(
            sigmas=[5.0, 180.0],
            means=[0, 300]
        )
        
        z = np.array([np.random.normal(means[l], sigmas[l]) for l in labels])
        self._save("fractal_clouds", z, labels)
    def generate_interlocking_moons(self):
        """
        Two interlocking non-convex half-moons. 
        GMMs struggle because they try to fit elliptical 'blobs' to these shapes.
        """
        n_half = self.n_samples // 2
        
        # Upper Moon
        theta_upper = np.linspace(0, np.pi, n_half)
        x_upper = 10 * np.cos(theta_upper) + np.random.normal(0, 0.5, n_half)
        y_upper = 10 * np.sin(theta_upper) + np.random.normal(0, 0.5, n_half)
        
        # Lower Moon (shifted to interlock)
        theta_lower = np.linspace(0, np.pi, n_half)
        x_lower = 10 * np.cos(theta_lower) + 5
        y_lower = -10 * np.sin(theta_lower) + 5
        
        # Combine coordinates
        self.x = np.concatenate([x_upper, x_lower])
        self.y = np.concatenate([y_upper, y_lower])
        self.coords = np.column_stack((self.x, self.y))
        
        labels = np.concatenate([np.zeros(n_half), np.ones(n_half)])
        
        # Regime 0 (Upper): High Volatility, Regime 1 (Lower): Low Volatility
        sigmas, means = self._get_params(
            sigmas=[150.0, 0.5],
            means=[200, 0]
        )
        
        z = np.array([np.random.normal(means[int(l)], sigmas[int(l)]) for l in labels])
        self._save("interlocking_moons", z, labels)

if __name__ == "__main__":
    # Standard Behavior: All clusters share Mean=50 (Variance signal only)
    factory_std = SyntheticFactory(n_samples=6000, mean_shift=False)
    factory_std.generate_voronoi_sharp()
    factory_std.generate_variance_gradient()
    factory_std.generate_sparse_islands()
    factory_std.generate_spiral_volatility()
    factory_std.generate_density_variance_bias()

    # Enhanced Behavior: Starkly different means and variances for clarity
    factory_shift = SyntheticFactory(n_samples=6000, mean_shift=True)
    factory_shift.generate_concentric_donuts()
    factory_shift.generate_anisotropic_streaks()
    factory_shift.generate_checkerboard_regimes()
    factory_shift.generate_nested_clusters()
    factory_shift.generate_fractal_noise()
    
    # GMM challenging dataset
    factory_shift.generate_interlocking_moons()