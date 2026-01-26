import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Pathing to allow imports from Models
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.Clis.engine import Clis
from models.Clis.metrics.evaluation import ClisEvaluator

def run_benchmark():
    data_dir = "data"
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Get all simulated datasets
    data_files = [f for f in os.listdir(data_dir) if f.endswith('.npz')]
    vm = ClisEvaluator()

    for d_file in data_files:
        print(f"Running Experiment on: {d_file}")
        
        # Load Data
        loader = np.load(os.path.join(data_dir, d_file))
        df = pd.DataFrame({'x': loader['x'], 'y': loader['y'], 'z': loader['z']})
        true_labels = loader['labels']

        # Fit Model
        model = Clis(loss_metric="nll", strategies=("axis", "radial", "oblique"))
        model.fit(df)
        preds = model.predict(df)

        # Compute Metrics
        scores = vm.structural_scores(true_labels, preds)
        purity_data = vm.intersection_report(true_labels, preds)
        contrast = vm.variance_contrast(df, preds)

        # Plotting
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].scatter(df['x'], df['y'], c=true_labels, cmap='tab10', s=2, alpha=0.5)
        axes[0].set_title(f"Simulation: {d_file}")
        
        axes[1].scatter(df['x'], df['y'], c=preds, cmap='prism', s=2, alpha=0.5)
        axes[1].set_title(f"Discovered (ARI: {scores['ARI']:.2f})")

        plt.savefig(os.path.join(results_dir, f"plot_{d_file.replace('.npz', '.png')}"))
        plt.close()

        # Log Metrics to a text file
        with open(os.path.join(results_dir, "metrics_log.txt"), "a") as f:
            f.write(f"\nDataset: {d_file} | ARI: {scores['ARI']:.4f} | Contrast: {contrast:.2f}\n")
            f.write(f"Intersection Report: {purity_data}\n")

if __name__ == "__main__":
    run_benchmark()