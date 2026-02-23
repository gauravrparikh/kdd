"""
5D CLIS Experiment: Split on 3 dimensions, compare joint distribution on 2 dimensions.

Setting:
- Split space: d0, d1, d2 (first 3 dimensions)
- Target space: d3, d4 (next 2 dimensions) - joint distribution
- CLIS uses MMD for merging (joint distribution comparison)

Includes challenging datasets where GMM/KMeans fail vs CLIS:
- Spiral volatility: non-convex spiral in 3D
- Density bias: sphere vs outer, density-variance mismatch
- Concentric shells: 3D shells (not ellipsoids)
- Variance-only: same mean, different covariances
- Checkerboard: 3D grid with alternating regimes
"""

import os
import sys
import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from scipy.spatial import cKDTree

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.Clis.engine import Clis
from models.Clis.clis_forest import ClisForest
from models.Clis.metrics.evaluation import ClisEvaluator


def generate_5d_voronoi(n_samples=5000, n_clusters=4, seed=42):
    """
    Generate 5D synthetic data:
    - d0, d1, d2: split space (3D) - used for partitioning
    - d3, d4: target space (2D) - joint distribution varies by cluster

    Clusters are defined by Voronoi in (d0,d1,d2). Each cluster has a distinct
    bivariate Gaussian for (d3, d4) with different means and covariances.
    """
    rng = np.random.default_rng(seed)

    # Split space: uniform in 3D cube
    split_data = rng.uniform(-10, 10, (n_samples, 3))

    # Cluster seeds in split space (Voronoi)
    seeds = rng.uniform(-8, 8, (n_clusters, 3))

    # Assign each point to nearest seed
    tree = cKDTree(seeds)
    _, labels = tree.query(split_data, k=1)

    # Target space: different bivariate Gaussian per cluster
    # (mean, cov) for each cluster
    cluster_params = [
        (np.array([0, 0]), np.array([[1, 0.2], [0.2, 1]])),
        (np.array([5, 5]), np.array([[3, 0.5], [0.5, 2]])),
        (np.array([-5, 3]), np.array([[0.5, 0], [0, 2]])),
        (np.array([2, -4]), np.array([[2, -0.8], [-0.8, 1.5]])),
    ]
    # Extend if n_clusters > 4
    while len(cluster_params) < n_clusters:
        cluster_params.append(
            (rng.uniform(-5, 5, 2), np.diag(rng.uniform(0.5, 3, 2)))
        )

    target_data = np.zeros((n_samples, 2))
    for k in range(n_clusters):
        mask = labels == k
        mu, cov = cluster_params[k]
        target_data[mask] = rng.multivariate_normal(mu, cov, size=mask.sum())

    # Full 5D data
    X_split = pd.DataFrame(
        split_data,
        columns=['d0', 'd1', 'd2']
    )
    y_target = target_data  # shape (n_samples, 2)

    return X_split, y_target, labels.flatten()


def generate_5d_spiral_volatility(n_samples=3000, seed=42):
    """
    Spiral in (d0,d1) plane with d2; inside spiral vs outside have different (d3,d4).
    Non-convex shape - GMM/KMeans struggle.
    """
    rng = np.random.default_rng(seed)
    # Uniform sample in 3D cube
    split_data = rng.uniform(-15, 15, (n_samples, 3))
    # Dense spiral curve in (d0,d1) plane
    theta_curve = np.linspace(0, 4 * np.pi, 500)
    r_curve = theta_curve * 1.2
    spiral_curve = np.column_stack([
        r_curve * np.cos(theta_curve),
        r_curve * np.sin(theta_curve),
        np.linspace(-5, 5, 500)
    ])
    tree = cKDTree(spiral_curve)
    dist_to_spiral, _ = tree.query(split_data, k=1)
    labels = (dist_to_spiral.flatten() < 3.0).astype(int)

    target_data = np.zeros((n_samples, 2))
    target_data[labels == 0] = rng.multivariate_normal([0, 0], [[2, 0.3], [0.3, 2]], size=(labels == 0).sum())
    target_data[labels == 1] = rng.multivariate_normal([0, 0], [[80, 10], [10, 60]], size=(labels == 1).sum())

    X_split = pd.DataFrame(split_data, columns=['d0', 'd1', 'd2'])
    return X_split, target_data, labels


def generate_5d_density_bias(n_samples=3000, seed=42):
    """
    Sphere (core) vs outer region. Core: dense, low var in (d3,d4). Outer: sparse, high var.
    Density-variance mismatch - GMM struggles.
    """
    rng = np.random.default_rng(seed)
    split_data = rng.uniform(-10, 10, (n_samples, 3))
    dist = np.sqrt(np.sum(split_data ** 2, axis=1))
    labels = (dist < 5).astype(int)

    target_data = np.zeros((n_samples, 2))
    target_data[labels == 0] = rng.multivariate_normal([0, 0], [[150, 20], [20, 120]], size=(labels == 0).sum())
    target_data[labels == 1] = rng.multivariate_normal([0, 0], [[1, 0.1], [0.1, 1]], size=(labels == 1).sum())

    X_split = pd.DataFrame(split_data, columns=['d0', 'd1', 'd2'])
    return X_split, target_data, labels


def generate_5d_concentric_shells(n_samples=3000, seed=42):
    """
    Three concentric shells in 3D. GMM fits ellipsoids, not shells.
    """
    rng = np.random.default_rng(seed)
    phi = rng.uniform(0, 2 * np.pi, n_samples)
    theta = np.arccos(rng.uniform(-1, 1, n_samples))
    r = np.zeros(n_samples)
    labels = np.zeros(n_samples, dtype=int)
    n_per = n_samples // 3
    r[:n_per] = rng.uniform(2, 4, n_per)
    labels[:n_per] = 0
    r[n_per:2*n_per] = rng.uniform(6, 8, n_per)
    labels[n_per:2*n_per] = 1
    n_rest = n_samples - 2 * n_per
    r[2*n_per:] = rng.uniform(10, 12, n_rest)
    labels[2*n_per:] = 2

    d0 = r * np.sin(theta) * np.cos(phi)
    d1 = r * np.sin(theta) * np.sin(phi)
    d2 = r * np.cos(theta)
    split_data = np.column_stack([d0, d1, d2])

    target_data = np.zeros((n_samples, 2))
    for k in range(3):
        mask = labels == k
        n = int(mask.sum())
        mu = [[0, 0], [3, 3], [-2, 2]][k]
        cov = [[[1, 0.2], [0.2, 1]], [[4, 1], [1, 3]], [[2, -0.5], [-0.5, 2]]][k]
        target_data[mask] = rng.multivariate_normal(mu, cov, size=n)

    X_split = pd.DataFrame(split_data, columns=['d0', 'd1', 'd2'])
    return X_split, target_data, labels


def generate_5d_variance_only(n_samples=3000, n_clusters=4, seed=42):
    """
    Same mean in (d3,d4) for all clusters, different covariances.
    GMM/KMeans cluster by mean - they struggle when means overlap.
    """
    rng = np.random.default_rng(seed)
    split_data = rng.uniform(-10, 10, (n_samples, 3))
    seeds = rng.uniform(-6, 6, (n_clusters, 3))
    tree = cKDTree(seeds)
    _, labels = tree.query(split_data, k=1)
    labels = labels.flatten()

    # Same mean (0,0), different covariances
    covs = [
        np.array([[1, 0.2], [0.2, 1]]),
        np.array([[15, 3], [3, 12]]),
        np.array([[0.5, 0], [0, 8]]),
        np.array([[6, -2], [-2, 4]]),
    ]
    while len(covs) < n_clusters:
        covs.append(np.diag(rng.uniform(0.5, 5, 2)))

    target_data = np.zeros((n_samples, 2))
    for k in range(n_clusters):
        mask = labels == k
        target_data[mask] = rng.multivariate_normal([0, 0], covs[k % len(covs)], size=mask.sum())

    X_split = pd.DataFrame(split_data, columns=['d0', 'd1', 'd2'])
    return X_split, target_data, labels


def generate_5d_checkerboard(n_samples=3000, seed=42):
    """
    3D grid; alternating cells have different (d3,d4) regimes.
    Sharp boundaries - GMM's ellipses struggle.
    """
    rng = np.random.default_rng(seed)
    split_data = rng.uniform(-12, 12, (n_samples, 3))
    grid_size = 4.0
    x_bins = ((split_data[:, 0] + 12) // grid_size).astype(int)
    y_bins = ((split_data[:, 1] + 12) // grid_size).astype(int)
    z_bins = ((split_data[:, 2] + 12) // grid_size).astype(int)
    labels = (x_bins + y_bins + z_bins) % 2

    target_data = np.zeros((n_samples, 2))
    target_data[labels == 0] = rng.multivariate_normal([0, 0], [[0.5, 0], [0, 0.5]], size=(labels == 0).sum())
    target_data[labels == 1] = rng.multivariate_normal([0, 0], [[80, 15], [15, 60]], size=(labels == 1).sum())

    X_split = pd.DataFrame(split_data, columns=['d0', 'd1', 'd2'])
    return X_split, target_data, labels


DATASET_GENERATORS = {
    "voronoi": (generate_5d_voronoi, 4),
    "spiral_volatility": (generate_5d_spiral_volatility, 2),
    "density_bias": (generate_5d_density_bias, 2),
    "concentric_shells": (generate_5d_concentric_shells, 3),
    "variance_only": (generate_5d_variance_only, 4),
    "checkerboard": (generate_5d_checkerboard, 2),
}


def run_5d_experiment():
    results_dir = os.path.join(os.path.dirname(__file__), "..", "results", "experiment_5d")
    os.makedirs(results_dir, exist_ok=True)

    print("=" * 60)
    print("5D CLIS Experiment: Split on 3 dims, Joint target on 2 dims")
    print("=" * 60)

    n_samples = 2000
    evaluator = ClisEvaluator()
    all_metrics = []

    for dataset_name, (gen_fn, n_clusters) in DATASET_GENERATORS.items():
        print(f"\n{'='*50}")
        print(f"Dataset: {dataset_name} (n_clusters={n_clusters})")
        print("=" * 50)

        # Generate data
        if dataset_name in ("voronoi", "variance_only"):
            X, y, true_labels = gen_fn(n_samples=n_samples, n_clusters=n_clusters)
        else:
            X, y, true_labels = gen_fn(n_samples=n_samples)

        # Train-test split
        X_train, X_test, y_train, y_test, labels_train, labels_test = train_test_split(
            X, y, true_labels, test_size=0.2, random_state=42
        )

        # Models (fresh per dataset)
        models = {
            "KMeans": KMeans(n_clusters=n_clusters, random_state=42),
            "GMM": GaussianMixture(n_components=n_clusters, random_state=42),
                "CLIS-Single": Clis(
                split_cols=['d0', 'd1', 'd2'],
                loss_metric="nll",
                complexity_penalty=0.01,
                lookahead_depth=0,
                merge_metric="mmd",
                merge_use_permutation=False,
                merge_mmd_threshold=0.15,
                min_samples_leaf=25,
            ),
            "CLIS-Forest": ClisForest(
                n_estimators=6,
                n_clusters=n_clusters,
                split_cols=['d0', 'd1', 'd2'],
                loss_metric="nll",
                complexity_penalty=0.01,
                lookahead_depth=0,
                merge_metric="mmd",
                merge_use_permutation=False,
            merge_mmd_threshold=0.15,
            min_samples_leaf=25,
        ),
        }

        for model_name, model in models.items():
            print(f"  Fitting {model_name}...", end=" ")
            start = time.time()
            if "CLIS" in model_name:
                model.fit(X_train, y_train)
                train_preds = model.predict(X_train)
                test_preds = model.predict(X_test)
            else:
                feat_train = np.column_stack([X_train, y_train])
                feat_test = np.column_stack([X_test, y_test])
                model.fit(feat_train)
                train_preds = model.predict(feat_train)
                test_preds = model.predict(feat_test)
            elapsed = time.time() - start

            ari_test = evaluator.structural_scores(labels_test, test_preds)['ARI']
            nmi_test = evaluator.structural_scores(labels_test, test_preds)['NMI']
            print(f"ARI={ari_test:.3f} NMI={nmi_test:.3f} Time={elapsed:.1f}s")

            all_metrics.append({
                "Dataset": dataset_name,
                "Model": model_name,
                "ARI_Test": ari_test,
                "NMI_Test": nmi_test,
                "Time_s": elapsed,
            })

        # Visualization per dataset
        try:
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(2, 3, figsize=(14, 10))
            axes[0, 0].scatter(X['d0'], X['d1'], c=true_labels, cmap='tab10', s=3, alpha=0.7)
            axes[0, 0].set_title(f"{dataset_name} - Ground Truth (d0,d1)")
            axes[0, 1].scatter(y[:, 0], y[:, 1], c=true_labels, cmap='tab10', s=3, alpha=0.7)
            axes[0, 1].set_title("Ground Truth (d3,d4)")

            for i, (mname, m) in enumerate(models.items()):
                if "CLIS" in mname:
                    preds = m.predict(X)
                else:
                    preds = m.predict(np.column_stack([X, y]))
                ari = next(r["ARI_Test"] for r in all_metrics if r["Dataset"] == dataset_name and r["Model"] == mname)
                ax = axes[(i + 2) // 3, (i + 2) % 3]
                ax.scatter(X['d0'], X['d1'], c=preds, cmap='tab10', s=3, alpha=0.7)
                ax.set_title(f"{mname} ARI={ari:.2f}")

            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, f"5d_{dataset_name}.png"), dpi=120)
            plt.close()
        except Exception as e:
            print(f"  Viz skipped: {e}")

    # Save aggregated results
    df = pd.DataFrame(all_metrics)
    out_path = os.path.join(results_dir, "metrics_5d.csv")
    df.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")
    print("\nSummary by dataset:")
    for ds in df["Dataset"].unique():
        sub = df[df["Dataset"] == ds]
        print(f"  {ds}: " + " | ".join(f"{r['Model']} ARI={r['ARI_Test']:.2f}" for _, r in sub.iterrows()))

    return all_metrics


if __name__ == "__main__":
    run_5d_experiment()
