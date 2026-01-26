import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

class ClisEvaluator:
    """Core evaluation suite for simulation-based partitioning."""

    @staticmethod
    def calculate_nll(z_values):
        """Gaussian Negative Log-Likelihood: Measures statistical fit."""
        n = len(z_values)
        if n < 2: return 0.0
        sigma = max(np.std(z_values), 1e-6)
        mu = np.mean(z_values)
        # Simplified NLL for relative comparison
        return n * np.log(sigma) + np.sum((z_values - mu)**2) / (2 * sigma**2)

    @staticmethod
    def intersection_report(true_labels, pred_labels):
        unique_preds = np.unique(pred_labels)
        report = {}
        for p_lab in unique_preds:
            if p_lab == -1: continue
            mask = (pred_labels == p_lab)
            
            # CAST TO INT HERE to fix the TypeError
            intersecting_true = true_labels[mask].astype(int) 
            
            if len(intersecting_true) == 0: continue
            
            # Now bincount will work
            counts = np.bincount(intersecting_true)
            majority_true_lab = np.argmax(counts)
            intersection_size = counts[majority_true_lab]
            purity = intersection_size / len(intersecting_true)
            
            report[p_lab] = {
                "matched_truth": int(majority_true_lab),
                "intersection_size": int(intersection_size),
                "purity": float(purity)
            }
        return report

    @staticmethod
    def structural_scores(true_labels, pred_labels):
        """ARI and NMI: Measures how well the shapes match the simulation."""
        return {
            "ARI": adjusted_rand_score(true_labels, pred_labels),
            "NMI": normalized_mutual_info_score(true_labels, pred_labels)
        }

    @staticmethod
    def variance_contrast(df, pred_labels):
        """Separation Power: Variance of the discovered cluster variances."""
        cluster_vars = []
        for lab in np.unique(pred_labels):
            if lab == -1: continue
            cluster_vars.append(np.var(df.loc[pred_labels == lab, 'z']))
        return np.var(cluster_vars) if len(cluster_vars) > 1 else 0