import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score
import folium
from folium.plugins import HeatMap

# Pathing to allow imports from Models
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.Clis.engine import Clis
from models.Clis.clis_forest import ClisForest
from models.Clis.metrics.evaluation import ClisEvaluator


def run_final_benchmark():

    data_dir = "/usr/xtmp/gr90/Spatial/data/orange_nc_geo/parcels/"
    results_dir = "results/final_benchmarks"
    os.makedirs(results_dir, exist_ok=True)

    # ---- Load the new parcel dataset ----
    data_path = os.path.join(data_dir, "orange_landvalue_per_acre.npy")
    data = np.load(data_path)

    # Data format: [x, y, value_per_acre]
    X = pd.DataFrame({'x': data[:, 0], 'y': data[:, 1]})
    y = data[:, 2]

    # Choose number of clusters manually
    n_clusters = 5

    evaluator = ClisEvaluator()

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 1. Initialize Models
    models = {
        "KMeans": KMeans(n_clusters=n_clusters, random_state=42),
        "GMM": GaussianMixture(n_components=n_clusters, random_state=42),
        "CLIS-Single": Clis(loss_metric="pinball", lookahead_depth=0),
        "CLIS-Forest": ClisForest(n_estimators=10, loss_metric="pinball", lookahead_depth=0),
    }

    results = {}

    for name, model in models.items():
        print(f"Fitting {name}...")
        start_fit = time.time()

        if "CLIS" in name:
            model.fit(X_train, y_train)
            train_preds = model.predict(X_train)
            test_preds = model.predict(X_test)
        else:
            feat_train = np.column_stack([X_train, y_train])
            feat_test = np.column_stack([X_test, y_test])
            model.fit(feat_train)
            train_preds = model.predict(feat_train)
            test_preds = model.predict(feat_test)

        fit_time = time.time() - start_fit

        results[name] = {
            "train": train_preds,
            "test": test_preds,
            "time": fit_time
        }

    # 2. Visualization (3D)
    fig = plt.figure(figsize=(25, 12))

    def add_3d_subplot(pos, title):
        ax = fig.add_subplot(2, 5, pos, projection='3d')
        ax.set_title(title)
        return ax

    # Raw value surface
    ax0 = add_3d_subplot(1, "Land Value Per Acre")
    sc = ax0.scatter(X['x'], X['y'], y, c=y, cmap='viridis', s=2)
    fig.colorbar(sc, ax=ax0, shrink=0.5)

    model_names = ["KMeans", "GMM", "CLIS-Single", "CLIS-Forest"]

    for i, name in enumerate(model_names):

        # Train
        ax_train = add_3d_subplot(i + 2, f"{name} (Train)\nTime: {results[name]['time']:.2f}s")
        ax_train.scatter(
            X_train['x'], X_train['y'], y_train,
            c=results[name]["train"], cmap='viridis', s=2
        )

        # Test
        ax_test = add_3d_subplot(i + 6, f"{name} (Test)")
        ax_test.scatter(
            X_test['x'], X_test['y'], y_test,
            c=results[name]["test"], cmap='viridis', s=5
        )

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "final_eval_3d_parcels.png"))
    plt.close()

    print("Finished benchmark.")
    fig2d, axes2d = plt.subplots(1, 4, figsize=(18, 5))
    for i, name in enumerate(model_names):
        ax = axes2d[i]
        ax.scatter(X_test['x'], X_test['y'], c=results[name]["test"], cmap='viridis', s=10, alpha=0.6)
        ax.set_title(f"{name} 2D Spatial Clusters")
        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "final_eval_2d_clusters.png"))

    # # ---- 3. NEW: Folium Map for CLIS ----
    # # Chapel Hill Coordinates
    # CH_LAT, CH_LON = 35.9132, -79.0558
    
    # # Create the base map
    # m = folium.Map(location=[CH_LAT, CH_LON], zoom_start=13, tiles='CartoDB positron')

    # lats, lons = X_test['x'], X_test['y']
    
    # heat_data = [[l, lo, val] for l, lo, val in zip(lats, lons, results["CLIS-Single"]["test"])]
    # HeatMap(heat_data, radius=15, blur=10, min_opacity=0.5).add_to(m)

    # # Save map
    # map_path = os.path.join(results_dir, "clis_chapel_hill_map.html")
    # m.save(map_path)
    # print(f"Folium map saved to {map_path}")


if __name__ == "__main__":
    run_final_benchmark()
