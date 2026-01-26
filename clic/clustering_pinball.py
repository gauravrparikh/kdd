import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.spatial import cKDTree

# Suppress warnings
warnings.filterwarnings('ignore')
np.random.seed(42)

# ==========================================
# 1. EXPANDED DATA GENERATION SUITE (10 SCENARIOS)
# ==========================================
print("Initializing 10 Simulation Scenarios...")

def gen_baseline_blobs(n=3000):
    """1. Baseline: Simple isotropic blobs with equal variance."""
    X, Y, Z, L = [], [], [], []
    centers = [[-5,-5], [5,5], [5,-5], [-5,5], [0,0]]
    for i, c in enumerate(centers):
        n_c = n // 5
        pts = np.random.normal(c, 1.5, size=(n_c, 2))
        X.extend(pts[:,0]); Y.extend(pts[:,1])
        # Z has moderate variance, distinct means
        Z.extend(np.random.normal(10 * i, 5.0, n_c)) 
        L.extend([i]*n_c)
    return np.array(X), np.array(Y), np.array(Z), np.array(L)

def gen_sharp_variance_voronoi(n=3000):
    """2. Sharp Variance: The original Voronoi case with drastic Z-variance differences."""
    x = np.random.uniform(-15, 15, n)
    y = np.random.uniform(-15, 15, n)
    points = np.column_stack((x, y))
    seeds = np.array([[-8, -8], [5, -5], [0, 8], [10, 10], [-10, 5]])
    tree = cKDTree(seeds)
    dists, labels = tree.query(points)
    
    z_sigmas = [0.1, 5.0, 20.0, 100.0, 200.0]
    z = np.zeros(n)
    for i in range(5):
        mask = labels == i
        if np.sum(mask) > 0:
            z[mask] = np.random.normal(50, z_sigmas[i], np.sum(mask))
    return x, y, z, labels

def gen_extreme_density_imbalance(n=3000):
    """3. Density Imbalance: Variable sample sizes (Dense vs Sparse clusters)."""
    X, Y, Z, L = [], [], [], []
    props = [0.02, 0.03, 0.05, 0.1, 0.8] # One massive cluster, tiny outliers
    centers = [[-8,-8], [8,8], [-8,8], [8,-8], [0,0]]
    
    for i, p in enumerate(props):
        n_c = int(n * p)
        pts = np.random.normal(centers[i], 1.0, size=(n_c, 2))
        X.extend(pts[:,0]); Y.extend(pts[:,1])
        Z.extend(np.random.normal(50, 10.0, n_c)) # Equal Z variance
        L.extend([i]*n_c)
    return np.array(X), np.array(Y), np.array(Z), np.array(L)

def gen_extreme_variance_imbalance(n=3000):
    """4. Variance Imbalance: Z-variances differ by orders of magnitude (Heteroscedasticity)."""
    X, Y, Z, L = [], [], [], []
    centers = [[-6,0], [-3,0], [0,0], [3,0], [6,0]]
    sigmas = [0.1, 1.0, 10.0, 50.0, 100.0] # Increasing variance
    
    for i, c in enumerate(centers):
        n_c = n // 5
        pts = np.random.normal(c, 0.8, size=(n_c, 2))
        X.extend(pts[:,0]); Y.extend(pts[:,1])
        Z.extend(np.random.normal(50, sigmas[i], n_c))
        L.extend([i]*n_c)
    return np.array(X), np.array(Y), np.array(Z), np.array(L)

def gen_anisotropic_clusters(n=3000):
    """5. Anisotropic: Stretched clusters (correlated X/Y) with different Z behavior."""
    X, Y, Z, L = [], [], [], []
    means = [[0,0], [5,5], [-5,-5]]
    covs = [ [[5, 4], [4, 5]], [[1, -0.8], [-0.8, 1]], [[2, 0], [0, 10]] ]
    
    for i in range(3):
        n_c = n // 3
        pts = np.random.multivariate_normal(means[i], covs[i], n_c)
        X.extend(pts[:,0]); Y.extend(pts[:,1])
        Z.extend(np.random.normal(10*i, 5.0, n_c))
        L.extend([i]*n_c)
    return np.array(X), np.array(Y), np.array(Z), np.array(L)

def gen_radial_heteroscedasticity(n=3000):
    """6. Radial Gradient: Variance increases with distance from center."""
    x = np.random.uniform(-10, 10, n)
    y = np.random.uniform(-10, 10, n)
    r = np.sqrt(x**2 + y**2)
    
    # Define shells
    labels = np.zeros(n, dtype=int)
    labels[r < 3] = 0
    labels[(r >= 3) & (r < 6)] = 1
    labels[r >= 6] = 2
    
    # Variance depends on radius
    z = np.random.normal(50, 1 + r**2, n) # Quadratic variance growth
    return x, y, z, labels

def gen_linear_gradient_variance(n=3000):
    """7. Linear Gradient: Continuous variance change from Left to Right."""
    x = np.random.uniform(0, 20, n)
    y = np.random.uniform(-5, 5, n)
    
    # 3 vertical strips
    labels = np.zeros(n, dtype=int)
    labels[x < 7] = 0
    labels[(x >= 7) & (x < 14)] = 1
    labels[x >= 14] = 2
    
    # Variance = x coordinate
    z = np.array([np.random.normal(50, max(0.1, val)) for val in x])
    return x, y, z, labels

def gen_heavy_tailed_noise(n=3000):
    """8. Heavy Tails: Laplace/Cauchy noise instead of Gaussian (Outliers)."""
    X, Y, Z, L = [], [], [], []
    centers = [[-5,0], [5,0]]
    for i, c in enumerate(centers):
        n_c = n // 2
        pts = np.random.normal(c, 1.5, size=(n_c, 2))
        X.extend(pts[:,0]); Y.extend(pts[:,1])
        if i == 0:
            # Gaussian (Clean)
            Z.extend(np.random.normal(50, 5, n_c))
        else:
            # Laplace (Heavy tails / Outliers)
            Z.extend(np.random.laplace(50, 5, n_c))
        L.extend([i]*n_c)
    return np.array(X), np.array(Y), np.array(Z), np.array(L)

def gen_checkerboard(n=3000):
    """9. Checkerboard: Grid structure with alternating high/low variance."""
    x = np.random.uniform(0, 10, n)
    y = np.random.uniform(0, 10, n)
    
    # Grid 2x2
    labels = np.zeros(n, dtype=int)
    mask_bl = (x < 5) & (y < 5)
    mask_br = (x >= 5) & (y < 5)
    mask_tl = (x < 5) & (y >= 5)
    mask_tr = (x >= 5) & (y >= 5)
    
    labels[mask_bl] = 0; labels[mask_br] = 1
    labels[mask_tl] = 2; labels[mask_tr] = 3
    
    z = np.zeros(n)
    z[mask_bl] = np.random.normal(50, 1.0, np.sum(mask_bl))   # Low Var
    z[mask_tr] = np.random.normal(50, 1.0, np.sum(mask_tr))   # Low Var
    z[mask_br] = np.random.normal(50, 20.0, np.sum(mask_br))  # High Var
    z[mask_tl] = np.random.normal(50, 20.0, np.sum(mask_tl))  # High Var
    
    return x, y, z, labels

def gen_concentric_rings(n=3000):
    """10. Concentric Rings: Non-linear separation."""
    theta = np.random.uniform(0, 2*np.pi, n)
    
    # Ring 1: R ~ N(3, 0.5)
    r1 = np.random.normal(3, 0.5, n//2)
    x1, y1 = r1 * np.cos(theta[:n//2]), r1 * np.sin(theta[:n//2])
    z1 = np.random.normal(50, 2, n//2)
    l1 = np.zeros(n//2)
    
    # Ring 2: R ~ N(8, 0.5)
    r2 = np.random.normal(8, 0.5, n//2)
    x2, y2 = r2 * np.cos(theta[n//2:]), r2 * np.sin(theta[n//2:])
    z2 = np.random.normal(50, 15, n//2)
    l2 = np.ones(n//2)
    
    return np.concatenate([x1,x2]), np.concatenate([y1,y2]), np.concatenate([z1,z2]), np.concatenate([l1,l2])

datasets = {
    '1. Baseline Blobs': gen_baseline_blobs,
    '2. Sharp Voronoi': gen_sharp_variance_voronoi,
    '3. Density Imbalance': gen_extreme_density_imbalance,
    '4. Variance Imbalance': gen_extreme_variance_imbalance,
    '5. Anisotropic': gen_anisotropic_clusters,
    '6. Radial Heteroscedasticity': gen_radial_heteroscedasticity,
    '7. Linear Gradient': gen_linear_gradient_variance,
    '8. Heavy Tails': gen_heavy_tailed_noise,
    '9. Checkerboard': gen_checkerboard,
    '10. Concentric Rings': gen_concentric_rings,
}

# ==========================================
# 2. ENHANCED PARTITIONING ALGORITHM
# ==========================================
class EnhancedSpatialPartitioner:
    def __init__(
        self,
        min_samples_leaf=30,
        gain_threshold=0.01,
        loss_metric="nll",
        use_axis=True,
        use_radial=True,
        use_oblique=True,
        random_state=42,
    ):
        self.min_samples_leaf = min_samples_leaf
        self.gain_threshold = gain_threshold
        self.loss_metric = loss_metric
        self.random_state = random_state
        np.random.seed(random_state)
        
        self.split_types = []
        if use_axis: self.split_types.append("axis")
        if use_radial: self.split_types.append("radial")
        if use_oblique: self.split_types.append("oblique")
            
        self.tree_ = {} 
        self.leaf_labels_ = {} 

    def calculate_loss(self, y):
        n = len(y)
        if n < 2: return 0.0
        if self.loss_metric == "mse":
            return np.sum((y - np.mean(y))**2)
        elif self.loss_metric == "nll":
            var = np.var(y)
            if var < 1e-6: var = 1e-6
            return (n / 2) * np.log(var) + (n / 2)
        elif self.loss_metric == "pinball":
            loss = 0.0
            for q in [0.1, 0.5, 0.9]:
                pred = np.percentile(y, q * 100)
                resid = y - pred
                loss += np.sum(np.maximum(q * resid, (q - 1) * resid))
            return loss
        return 0.0

    def propose_split(self, sub_data):
        stype = np.random.choice(self.split_types)
        if stype == "axis":
            axis = np.random.choice(["x", "y"])
            lo, hi = sub_data[axis].min(), sub_data[axis].max()
            if hi - lo < 1e-5: return None
            val = np.random.uniform(lo, hi)
            return ("axis", {"axis": axis, "value": val})
        elif stype == "radial":
            cx = np.random.uniform(sub_data["x"].min(), sub_data["x"].max())
            cy = np.random.uniform(sub_data["y"].min(), sub_data["y"].max())
            dists = np.sqrt((sub_data["x"]-cx)**2 + (sub_data["y"]-cy)**2)
            if dists.max() < 1e-5: return None
            r = np.random.uniform(0, dists.max())
            return ("radial", {"cx": cx, "cy": cy, "r": r})
        elif stype == "oblique":
            theta = np.random.uniform(0, 2*np.pi)
            a, b = np.cos(theta), np.sin(theta)
            proj = a * sub_data["x"] + b * sub_data["y"]
            c = np.random.uniform(proj.min(), proj.max())
            return ("oblique", {"a": a, "b": b, "c": c})
        return None

    def apply_split(self, indices, data, split_info):
        sub = data.iloc[indices]
        stype, params = split_info
        
        if stype == "axis":
            mask = sub[params["axis"]] < params["value"]
        elif stype == "radial":
            dists = np.sqrt((sub["x"] - params["cx"])**2 + (sub["y"] - params["cy"])**2)
            mask = dists < params["r"]
        elif stype == "oblique":
            val = params["a"] * sub["x"] + params["b"] * sub["y"]
            mask = val < params["c"]
            
        left = np.array(indices)[mask]
        right = np.array(indices)[~mask]
        return left.tolist(), right.tolist()

    def find_best_split(self, data, indices):
        sub_data = data.iloc[indices]
        current_y = sub_data["z"].values
        parent_loss = self.calculate_loss(current_y)
        
        best_gain = -np.inf
        best_split_info = None
        best_children = None
        
        n_attempts = 40 if len(self.split_types) > 1 else 15
        
        for _ in range(n_attempts):
            split_proposal = self.propose_split(sub_data)
            if split_proposal is None: continue
            
            left, right = self.apply_split(indices, data, split_proposal)
            if len(left) < self.min_samples_leaf or len(right) < self.min_samples_leaf:
                continue
            
            y_l = data.iloc[left]["z"].values
            y_r = data.iloc[right]["z"].values
            
            gain = parent_loss - (self.calculate_loss(y_l) + self.calculate_loss(y_r))
            
            if gain > best_gain:
                best_gain = gain
                best_split_info = split_proposal
                best_children = (left, right)
                
        return best_split_info, best_gain, best_children

    def fit(self, data):
        self.tree_ = {} 
        self.leaf_labels_ = {}
        
        n = len(data)
        partitions = {0: list(range(n))} 
        
        initial_loss = self.calculate_loss(data["z"].values)
        eff_threshold = self.gain_threshold * abs(initial_loss) if initial_loss != 0 else 0.001
        
        queue = [0]
        next_node_id = 1
        
        while queue:
            pid = queue.pop(0)
            indices = partitions[pid]
            
            if len(indices) >= 2 * self.min_samples_leaf:
                split_info, gain, children = self.find_best_split(data, indices)
                
                if split_info is not None and gain >= eff_threshold:
                    left_indices, right_indices = children
                    left_id = next_node_id
                    right_id = next_node_id + 1
                    next_node_id += 2
                    
                    self.tree_[pid] = (split_info, left_id, right_id)
                    partitions[left_id] = left_indices
                    partitions[right_id] = right_indices
                    queue.append(left_id)
                    queue.append(right_id)
                    continue
            
            self.leaf_labels_[pid] = pid 
            
        return self

    def predict(self, data):
        n = len(data)
        labels = np.zeros(n, dtype=int)
        
        for i in range(n):
            row = data.iloc[i]
            node = 0
            while node in self.tree_:
                split_info, left, right = self.tree_[node]
                stype, params = split_info
                
                go_left = False
                if stype == "axis":
                    go_left = row[params["axis"]] < params["value"]
                elif stype == "radial":
                    dist = np.sqrt((row["x"] - params["cx"])**2 + (row["y"] - params["cy"])**2)
                    go_left = dist < params["r"]
                elif stype == "oblique":
                    val = params["a"] * row["x"] + params["b"] * row["y"]
                    go_left = val < params["c"]
                
                node = left if go_left else right
            
            labels[i] = self.leaf_labels_.get(node, -1)
            
        return labels

# ==========================================
# 3. EVALUATION LOGIC
# ==========================================
def evaluate_metric(test_data, labels, method_name):
    unique = np.unique(labels)
    total_nll = 0
    test_y = test_data['z'].values
    
    for k in unique:
        if k == -1: continue 
        mask = labels == k
        z_sub = test_y[mask]
        if len(z_sub) < 2: continue
            
        sigma = np.std(z_sub)
        mu = np.mean(z_sub)
        if sigma < 1e-5: sigma = 1e-5
        
        term = len(z_sub) * np.log(sigma) + np.sum((z_sub - mu)**2)/(2*sigma**2)
        total_nll += term
    
    nll_score = total_nll / len(test_data)
    return nll_score

# ==========================================
# 4. MAIN LOOP OVER ALL 10 DATASETS
# ==========================================
final_summary = []

for ds_name, gen_func in datasets.items():
    print(f"\nProcessing: {ds_name} ...")
    
    # Generate Data
    x, y, z, true_labels = gen_func(2000)
    df = pd.DataFrame({'x': x, 'y': y, 'z': z})
    
    train_df, test_df, train_labels, test_labels = train_test_split(
        df, true_labels, test_size=0.4, random_state=42
    )
    
    # 1. GMM Baseline
    gmm = GaussianMixture(n_components=5, random_state=42)
    # GMM sees X, Y, Z
    X_all = StandardScaler().fit_transform(train_df[['x', 'y', 'z']])
    X_test_all = StandardScaler().fit_transform(test_df[['x', 'y', 'z']])
    gmm.fit(X_all)
    gmm_pred = gmm.predict(X_test_all)
    nll_gmm = evaluate_metric(test_df, gmm_pred, "GMM")
    
    # 2. Enhanced Partitioning (NLL)
    sp = EnhancedSpatialPartitioner(loss_metric="nll", use_axis=True, use_radial=True, use_oblique=True)
    sp.fit(train_df)
    sp_pred = sp.predict(test_df)
    nll_sp = evaluate_metric(test_df, sp_pred, "Enhanced SP")
    
    final_summary.append({
        'Dataset': ds_name,
        'GMM_NLL': nll_gmm,
        'SP_NLL': nll_sp,
        'Improvement': nll_gmm - nll_sp
    })
    
    # 3. PLOTTING (ALL DATASETS)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Ground Truth
    axes[0].scatter(test_df['x'], test_df['y'], c=test_labels, cmap='tab10', s=10, alpha=0.7)
    axes[0].set_title(f"{ds_name}\nTrue Labels")
    
    # GMM
    axes[1].scatter(test_df['x'], test_df['y'], c=gmm_pred, cmap='tab10', s=10, alpha=0.7)
    axes[1].set_title(f"GMM (NLL: {nll_gmm:.2f})")
    
    # Enhanced SP
    axes[2].scatter(test_df['x'], test_df['y'], c=sp_pred, cmap='tab10', s=10, alpha=0.7)
    axes[2].set_title(f"Enhanced SP (NLL: {nll_sp:.2f})")
    
    plt.tight_layout()
    # Create safe filename
    safe_name = ds_name.split(". ")[0] + "_" + ds_name.split(". ")[1].replace(" ", "_")
    filename = f"eval_{safe_name}.png"
    plt.savefig(filename)
    plt.close(fig) # Close figure to free memory
    print(f"  -> Saved plot: {filename}")

print("\n" + "="*60)
print("FINAL RESULTS SUMMARY (Sorted by Improvement)")
print("="*60)
res_df = pd.DataFrame(final_summary)
res_df = res_df.sort_values("Improvement", ascending=False)
print(res_df[['Dataset', 'GMM_NLL', 'SP_NLL', 'Improvement']].round(2).to_string(index=False))
print("="*60)