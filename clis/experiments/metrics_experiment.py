import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture


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
            "CLIS-Single": Clis(loss_metric="pinball", complexity_penalty=0.5),
            "CLIS-Forest": ClisForest(n_estimators=10, n_clusters=n_clusters, loss_metric="pinball", complexity_penalty=0.5)
        }

        results = {}
        for name, model in models.items():
            start_fit = time.time()
            if "CLIS" in name:
                model.fit(X_train, y_train)
                train_preds = model.predict(X_train)
                test_preds = model.predict(X_test)
            else:
                # GMM/KMeans use spatial + variance features
                feat_train = np.column_stack([X_train, y_train])
                feat_test = np.column_stack([X_test, y_test])
                model.fit(feat_train)
                train_preds = model.predict(feat_train)
                test_preds = model.predict(feat_test)
            
            fit_time = time.time() - start_fit

            scores = evaluator.structural_scores(labels_test, test_preds)
            
            leakage = evaluator.boundary_leakage_score(X_test, labels_test, test_preds)
            hinge_loss = evaluator.spatial_hinge_loss(X_test, labels_test, test_preds)            
            starkness = evaluator.boundary_variance_starkness(X_test, y_test, test_preds)
            
            results[name] = {
                "train": train_preds, 
                "test": test_preds, 
                "time": fit_time,
                "ARI": scores['ARI'],
                "Leakage": leakage,
                "Hinge": hinge_loss,
                "Starkness": starkness
            }

            all_metrics.append({
                "Dataset": d_file, 
                "Model": name, 
                "ARI_Test": scores['ARI'], 
                "Leakage": leakage,
                "Hinge_Loss": hinge_loss,
                "Starkness": starkness,
                "Fit_Time": fit_time
            })

        fig, axes = plt.subplots(2, 5, figsize=(25, 12))
        
        axes[0, 0].scatter(X['x'], X['y'], c=true_labels, cmap='tab10', s=2)
        axes[0, 0].set_title("Ground Truth Labels")
        
        sc = axes[1, 0].scatter(X['x'], X['y'], c=y, cmap='viridis', s=2)
        fig.colorbar(sc, ax=axes[1, 0])
        axes[1, 0].set_title("Variance Signal (Z)")

        model_names = ["KMeans", "GMM", "CLIS-Single", "CLIS-Forest"]
        for i, name in enumerate(model_names):
            axes[0, i+1].scatter(X_train['x'], X_train['y'], c=results[name]["train"], cmap='prism', s=2)
            axes[0, i+1].set_title(f"{name} (Train)\nTime: {results[name]['time']:.2f}s")
            
            res = results[name]
            axes[1, i+1].scatter(X_test['x'], X_test['y'], c=res["test"], cmap='prism', s=5)
            axes[1, i+1].set_title(
                f"{name} (Test)\nARI: {res['ARI']:.2f} | Leak: {res['Leakage']:.2f}\n"
                f"Hinge: {res['Hinge']:.2f} | Stark: {res['Starkness']:.1f}"
            )

        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f"final_eval_{d_file.replace('.npz', '.png')}"))
        plt.close()
    pd.DataFrame(all_metrics).to_csv(os.path.join(results_dir, "final_metrics_comprehensive.csv"), index=False)
    print(f"\n[SUCCESS] Comprehensive metrics saved to {results_dir}")

if __name__ == "__main__":
    run_final_benchmark()