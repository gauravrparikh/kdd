import numpy as np
import os
from scipy.spatial import cKDTree

class SyntheticFactory:
    def __init__(self, n_samples=5000, seed=42):
        self.n_samples = n_samples
        self.seed = seed
        np.random.seed(seed)
        # Pre-generate coordinates
        self.x = np.random.uniform(-15, 15, n_samples)
        self.y = np.random.uniform(-15, 15, n_samples)
        self.coords = np.column_stack((self.x, self.y))

    def _save(self, name, z, labels):
        output_path = os.path.join("", "data", f"{name}.npz")
        np.savez(output_path, x=self.x, y=self.y, z=z, labels=labels)
        print(f"Saved: {name}")

    def generate_voronoi_sharp(self):
        """Sharp boundaries based on proximity to seeds."""
        seeds = np.array([[-8, -8], [5, -5], [0, 8], [10, 10], [-10, 5]])
        tree = cKDTree(seeds)
        _, labels = tree.query(self.coords)
        z_sigmas = [0.1, 5.0, 50.0, 150.0, 300.0]
        z = np.array([np.random.normal(50, z_sigmas[l % 5]) for l in labels])
        self._save("voronoi_sharp", z, labels)

    def generate_variance_gradient(self):
        """Variance increases linearly from left to right (Continuous case)."""
        # labels here are just 'bins' for visualization purposes
        labels = (self.x - self.x.min()) / (self.x.max() - self.x.min()) * 5
        labels = labels.astype(int)
        # Sigma is a function of X
        sigmas = 10 + (self.x - self.x.min()) * 15 
        z = np.random.normal(50, sigmas)
        self._save("linear_gradient", z, labels)

    def generate_concentric_donuts(self):
        """Variance changes based on distance from the origin."""
        dist = np.sqrt(self.x**2 + self.y**2)
        # Create 3 zones: Inner stable, middle volatile, outer stable
        labels = np.zeros(self.n_samples)
        labels[(dist > 5) & (dist <= 10)] = 1
        labels[dist > 10] = 2
        
        sigmas = [2.0, 200.0, 10.0]
        z = np.array([np.random.normal(50, sigmas[int(l)]) for l in labels])
        self._save("concentric_donuts", z, labels)

    def generate_anisotropic_streaks(self):
        """Variance changes along an oblique 'stripe' pattern."""
        # Project points onto a 45-degree vector
        projection = (self.x + self.y) / np.sqrt(2)
        labels = (projection // 5).astype(int) % 2 # Alternating stripes
        
        sigmas = [1.0, 100.0]
        z = np.array([np.random.normal(50, sigmas[l]) for l in labels])
        self._save("oblique_stripes", z, labels)

    def generate_sparse_islands(self):
        """Small 'hotspots' of high variance in a sea of low variance."""
        labels = np.zeros(self.n_samples)
        hotspots = np.array([[-10, 10], [12, -2], [0, 0]])
        for hs in hotspots:
            dist = np.sqrt((self.x - hs[0])**2 + (self.y - hs[1])**2)
            labels[dist < 3] = 1 # Mark as hotspot
            
        sigmas = [5.0, 400.0]
        z = np.array([np.random.normal(50, sigmas[int(l)]) for l in labels])
        self._save("sparse_islands", z, labels)
        
    def generate_spiral_volatility(self):
        """Variance follows an Archimedean spiral. Challenges oblique/radial combinations."""
        theta = np.sqrt(np.random.rand(self.n_samples)) * 4 * np.pi 
        r = theta * 1.2
        # Transform spiral to our -15 to 15 coordinate space
        spiral_x = r * np.cos(theta)
        spiral_y = r * np.sin(theta)
        
        # We assign high variance to points close to the spiral line
        tree = cKDTree(np.column_stack((spiral_x, spiral_y)))
        dist, _ = tree.query(self.coords)
        
        labels = (dist < 2.5).astype(int)
        sigmas = [2.0, 250.0] # High volatility on the spiral path
        z = np.array([np.random.normal(50, sigmas[l]) for l in labels])
        self._save("spiral_volatility", z, labels)

    def generate_checkerboard_regimes(self):
        """Classic checkerboard variance. Tests axis-aligned precision."""
        # Create 5x5 grid
        grid_size = 6.0 
        x_bins = ((self.x + 15) // grid_size).astype(int)
        y_bins = ((self.y + 15) // grid_size).astype(int)
        
        labels = (x_bins + y_bins) % 2
        sigmas = [0.5, 120.0]
        z = np.array([np.random.normal(50, sigmas[l]) for l in labels])
        self._save("checkerboard", z, labels)

    def generate_density_variance_bias(self):
        """
        High variance in sparse regions, Low variance in dense regions. 
        Tests if the model is biased by point density.
        """
        # Create a dense 'core'
        core_mask = (np.sqrt(self.x**2 + self.y**2) < 7)
        labels = core_mask.astype(int)
        
        # Invert: Dense center has σ=2, Sparse outskirts have σ=300
        sigmas = [300.0, 2.0]
        z = np.array([np.random.normal(50, sigmas[l]) for l in labels])
        self._save("density_bias", z, labels)

    def generate_nested_clusters(self):
        """A high-variance cluster inside a low-variance cluster inside a high-variance sea."""
        dist = np.sqrt(self.x**2 + self.y**2)
        labels = np.zeros(self.n_samples)
        labels[dist < 10] = 1 # Mid ring
        labels[dist < 4] = 2  # Inner core
        
        sigmas = [200.0, 5.0, 200.0]
        z = np.array([np.random.normal(50, sigmas[int(l)]) for l in labels])
        self._save("nested_targets", z, labels)

    def generate_fractal_noise(self):
        """Uses a sum of sine waves to create 'clouds' of variance."""
        val = np.sin(self.x/3) * np.cos(self.y/3) + np.sin((self.x+self.y)/5)
        labels = (val > 0).astype(int)
        sigmas = [5.0, 180.0]
        z = np.array([np.random.normal(50, sigmas[l]) for l in labels])
        self._save("fractal_clouds", z, labels)

if __name__ == "__main__":
    factory = SyntheticFactory(n_samples=6000)
    factory.generate_voronoi_sharp()
    factory.generate_variance_gradient()
    factory.generate_concentric_donuts()
    factory.generate_anisotropic_streaks()
    factory.generate_sparse_islands()
    factory.generate_spiral_volatility()
    factory.generate_checkerboard_regimes()
    factory.generate_density_variance_bias()
    factory.generate_nested_clusters()
    factory.generate_fractal_noise()