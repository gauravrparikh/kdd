import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from scipy.spatial import cKDTree

class ClisEvaluator:
    """Core evaluation suite for simulation-based partitioning."""

    @staticmethod
    def calculate_nll(z_values):
        """Gaussian Negative Log-Likelihood: Measures statistical fit."""
        n = len(z_values)
        if n < 2: return 0.0
        sigma = max(np.std(z_values), 1e-6)
        mu = np.mean(z_values)
        return n * np.log(sigma) + np.sum((z_values - mu)**2) / (2 * sigma**2)

    @staticmethod
    def intersection_report(true_labels, pred_labels):
        unique_preds = np.unique(pred_labels)
        report = {}
        for p_lab in unique_preds:
            if p_lab == -1: continue
            mask = (pred_labels == p_lab)
            intersecting_true = true_labels[mask].astype(int) 
            if len(intersecting_true) == 0: continue
            
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

    @staticmethod
    def boundary_leakage_score(X, true_labels, pred_labels):
        """
        Measures the proportion of points that violate hard spatial boundaries.
        Favors hard geometric cuts over soft probabilistic overlap.
        """
        violations = 0
        unique_labels = np.unique(true_labels)
        for label in unique_labels:
            true_mask = (true_labels == label)
            if not np.any(true_mask): continue
            counts = np.bincount(pred_labels[true_mask].astype(int))
            major_pred_label = np.argmax(counts)
            violations += np.sum(pred_labels[true_mask] != major_pred_label)
        return violations / len(true_labels)

    @staticmethod
    def spatial_hinge_loss(X, true_labels, pred_labels):
        """
        SVM-inspired metric. Penalizes points based on their distance 
        from the 'correct' side of the boundary.
        """
        X_vals = X[['x', 'y']].values
        total_penalty = 0.0
        unique_labels = np.unique(true_labels)
        
        for label in unique_labels:
            misclassified_mask = (true_labels == label) & (pred_labels != label)
            correct_points_mask = (true_labels == label) & (pred_labels == label)
            
            if not np.any(misclassified_mask) or not np.any(correct_points_mask):
                continue
                
            tree = cKDTree(X_vals[correct_points_mask])
            distances, _ = tree.query(X_vals[misclassified_mask])
            total_penalty += np.sum(distances)
            
        return total_penalty / len(X)

    @staticmethod
    def boundary_variance_starkness(X, y, pred_labels, buffer_dist=1.0):
        """
        Measures the absolute difference in mean variance across discovered boundaries.
        Higher values indicate cleaner, sharper variance-based partitions.
        """
        X_vals = X[['x', 'y']].values
        gradients = []
        unique_labels = np.unique(pred_labels)
        
        for i in range(len(unique_labels)):
            for j in range(i + 1, len(unique_labels)):
                mask_i = (pred_labels == unique_labels[i])
                mask_j = (pred_labels == unique_labels[j])
                
                if not np.any(mask_i) or not np.any(mask_j): continue
                
                tree_j = cKDTree(X_vals[mask_j])
                dist_to_j, _ = tree_j.query(X_vals[mask_i])
                
                border_points_i = y[mask_i][dist_to_j < buffer_dist]
                border_points_j = y[mask_j][cKDTree(X_vals[mask_i]).query(X_vals[mask_j])[0] < buffer_dist]
                
                if len(border_points_i) > 0 and len(border_points_j) > 0:
                    grad = abs(np.mean(border_points_i) - np.mean(border_points_j))
                    gradients.append(grad)
        
        return np.mean(gradients) if gradients else 0.0