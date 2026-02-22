import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

# Pathing to allow imports from Models
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.Clis.engine import Clis
from models.Clis.clis_forest import ClisForest
from models.Clis.metrics.evaluation import ClisEvaluator

def run_final_benchmark():
    data_dir = "data"
    results_dir = "results/final_benchmarks"
    os.makedirs(results_dir, exist_ok=True)
    
    data_files = [f for f in os.listdir(data_dir) if f.endswith('.npz')]
    evaluator = ClisEvaluator()
    all_metrics = []

    for d_file in data_files:
        print(f"\n--- Processing: {d_file} ---")
        loader = np.load(os.path.join(data_dir, d_file))
        X = pd.DataFrame({'x': loader['x'], 'y': loader['y']})
        y = loader['z']
        true_labels = loader['labels']
        n_clusters = len(np.unique(true_labels))

        # Train-Test Split
        X_train, X_test, y_train, y_test, labels_train, labels_test = train_test_split(
            X, y, true_labels, test_size=0.2, random_state=42
        )

        # 1. Initialize Models
        models = {
            "KMeans": KMeans(n_clusters=n_clusters, random_state=42),
            "GMM": GaussianMixture(n_components=n_clusters, random_state=42),
            "CLIS-Single": Clis(loss_metric="pinball", lookahead_depth=0),
            # "CLIS-Forest": ClisForest(n_estimators=10, n_clusters=n_clusters, loss_metric="pinball", lookahead_depth=0)
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
            results[name] = {"train": train_preds, "test": test_preds, "time": fit_time}

            # 2. Record Comparative Metrics
            ari_test = evaluator.structural_scores(labels_test, test_preds)['ARI']
            all_metrics.append({
                "Dataset": d_file, "Model": name, "ARI_Test": ari_test, "Fit_Time": fit_time
            })

        # 3. Visualization (10 Panels: Ground Truth + Z + 4 Train + 4 Test)
        # Modified to use 3D projection
        fig = plt.figure(figsize=(25, 12))
        
        # Helper to create 3D axes
        def add_3d_subplot(pos, title):
            ax = fig.add_subplot(2, 5, pos, projection='3d')
            ax.set_title(title)
            return ax

        # Ground Truths
        ax00 = add_3d_subplot(1, "Ground Truth Labels")
        ax00.scatter(X['x'], X['y'], y, c=true_labels, cmap='tab10', s=2)
        
        ax10 = add_3d_subplot(6, "Variance Signal (Z)")
        sc = ax10.scatter(X['x'], X['y'], y, c=y, cmap='viridis', s=2)
        fig.colorbar(sc, ax=ax10, shrink=0.5)

        # Map Results to Plots
        model_names = ["KMeans", "GMM", "CLIS-Single"] # "CLIS-Forest"
        for i, name in enumerate(model_names):
            # Train Plots
            ax_train = add_3d_subplot(i+2, f"{name} (Train)\nTime: {results[name]['time']:.2f}s")
            ax_train.scatter(X_train['x'], X_train['y'], y_train, c=results[name]["train"], cmap='prism', s=2)
            
            # Test Plots
            ari = evaluator.structural_scores(labels_test, results[name]["test"])['ARI']
            ax_test = add_3d_subplot(i+7, f"{name} (Test)\nARI: {ari:.2f}")
            ax_test.scatter(X_test['x'], X_test['y'], y_test, c=results[name]["test"], cmap='prism', s=5)

        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f"final_eval_3d_{d_file.replace('.npz', '.png')}"))
        plt.close()

    # Save CSV metrics
    pd.DataFrame(all_metrics).to_csv(os.path.join(results_dir, "final_metrics_3d.csv"), index=False)

if __name__ == "__main__":
    run_final_benchmark()