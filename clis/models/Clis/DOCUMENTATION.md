# CLIS: Clustering with Loss-based Independence Splitting

## Overview

**CLIS** is a spatial clustering framework designed to discover **regime-based partitions**—regions of space where a target variable \(z\) exhibits distinct statistical properties (e.g., different variances, quantiles, or means). Unlike traditional spatial clustering (KMeans, GMM) that groups points by geometric proximity, CLIS partitions space based on **statistical heterogeneity** of the target variable across coordinates \((x, y)\).

### Problem Setting

- **Input**: Spatial coordinates \((x, y)\) and a target variable \(z\) (e.g., volatility, temperature, pollution levels)
- **Output**: Cluster labels that group locations with similar *statistical regimes* of \(z\)
- **Use case**: Identify regions where variance, risk, or volatility differs—e.g., high-variance hotspots vs. stable regions

---

## Architecture

```
ClisForest (ensemble)
    └── Clis (single tree) × n_estimators
            ├── Split Strategies (axis, radial, oblique, elliptical)
            ├── Loss Functions (MSE, NLL, pinball)
            └── Leaf Merging (Kolmogorov-Smirnov)
```

---

## Core Components

### 1. Clis (Single Tree) — `engine.py`

The **Clis** class implements a single decision tree that recursively partitions space to minimize a loss on the target variable \(z\).

#### Algorithm Flow

1. **Recursive Splitting**
   - Start with all data at the root.
   - For each node, propose candidate splits using multiple spatial strategies.
   - Choose the split that maximizes **gain** = parent loss − (left loss + right loss) − complexity penalty.
   - Continue until nodes are too small or no split improves the objective.

2. **Leaf Merging**
   - After splitting, leaves may be statistically similar (e.g., same variance).
   - Use the **Kolmogorov-Smirnov test** to compare distributions of \(z\) in each pair of leaves.
   - If \(p\)-value > `merge_threshold`, merge leaves (Union-Find).
   - Final clusters = merged leaf groups.

#### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `min_samples_leaf` | 10 | Minimum samples per leaf; prevents overfitting |
| `gain_threshold` | 0.001 | Minimum gain to accept a split |
| `loss_metric` | `"pinball"` | Loss: `"mse"`, `"nll"`, or `"pinball"` |
| `strategies` | axis, radial, oblique, elliptical | Spatial split strategies to use |
| `complexity_penalty` | 1.0 | Penalty for splitting (controls tree size) |
| `lookahead_depth` | 2 | Depth of lookahead for marginal splits |
| `merge_threshold` | 0.005 | KS p-value above which leaves are merged |
| `min_depth` | 1 | Minimum depth before early stopping |

#### Loss Functions

- **MSE**: \(\sum (z - \bar{z})^2\) — variance reduction
- **NLL**: Gaussian negative log-likelihood — fits Gaussian regimes
- **Pinball**: Quantile loss over \(q \in \{0.01, 0.05, \ldots, 0.99\}\) — robust to outliers and captures full distribution shape

---

### 2. Split Strategies — `split_strategies.py`

Four geometric split types define how space is divided. Each strategy has `propose()` (sample split parameters) and `apply()` (return boolean mask for left/right).

#### Axis-Aligned Split (`AxisSplit`)

- Splits along \(x\) or \(y\) at a threshold.
- Parameters: `axis`, `value`
- Condition: `data[axis] < value`

#### Radial Split (`RadialSplit`)

- Splits by distance from a center \((c_x, c_y)\).
- Parameters: `cx`, `cy`, `r`
- Condition: \(\sqrt{(x - c_x)^2 + (y - c_y)^2} < r\)
- Handles circular/ring-shaped regimes.

#### Oblique Split (`ObliqueSplit`)

- Splits by a linear projection \(a \cdot x + b \cdot y < c\).
- Parameters: `a`, `b`, `c` (angle and intercept)
- Handles diagonal stripes and anisotropic patterns.

#### Elliptical Split (`EllipticalSplit`)

- Splits by ellipse: \(\left(\frac{x'}{a}\right)^2 + \left(\frac{y'}{b}\right)^2 < 1\) after rotation.
- Parameters: `cx`, `cy`, `a`, `b`, `angle`
- Handles elongated, non-circular regions.

All strategies use **randomized proposals** (percentiles, jitter) to avoid trivial splits and break symmetry.

---

### 3. Lookahead and Gain Evaluation — `engine.py`

For each candidate split, the tree evaluates:

1. **Immediate gain**: parent loss − (left loss + right loss) − split penalty
2. **Lookahead** (optional): If gain is marginal, recursively evaluate best sub-splits on left and right; add potential future gains.
3. **Early exit**: If immediate gain is clearly best, skip expensive lookahead.
4. **Adaptive sampling**: 50 proposals at root, 10 at deeper nodes.

This balances exploration of complex splits with computational cost.

---

### 4. ClisForest (Ensemble) — `clis_forest.py`

An ensemble of Clis trees with a scalable consensus mechanism.

#### Training

- Bootstrap sampling: each tree trains on a random subset (e.g., 50% of data).
- Each tree is a full Clis model with its own splits and leaves.

#### Prediction (Scalable Consensus)

Instead of building an \(N \times N\) co-association matrix:

1. **Leaf embedding**: For each sample, collect leaf IDs from all trees → matrix of shape \((N, n\_estimators)\).
2. **Consensus clustering**: Run **MiniBatchKMeans** on this leaf matrix with `n_clusters` clusters.
3. **Output**: Cluster labels from KMeans.

This is \(O(N)\) instead of \(O(N^2)\) or \(O(N^3)\), making it scalable.

#### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_estimators` | 10 | Number of Clis trees |
| `bootstrap_sample_ratio` | 0.5 | Fraction of data per tree |
| `n_clusters` | 3 | Number of final clusters (KMeans) |

---

## Evaluation Metrics — `metrics/evaluation.py`

The `ClisEvaluator` class provides metrics for assessing partition quality:

| Metric | Description |
|--------|-------------|
| **ARI** | Adjusted Rand Index — agreement with ground truth |
| **NMI** | Normalized Mutual Information — information overlap |
| **Variance contrast** | Variance of cluster variances — separation of regimes |
| **Boundary leakage** | Proportion of points violating true boundaries |
| **Spatial hinge loss** | Distance-based penalty for misclassified points |
| **Boundary variance starkness** | Mean difference in variance across discovered boundaries |
| **Distribution continuity** | Mean KS statistic between cluster pairs |
| **NLL** | Negative log-likelihood of cluster fit |

---

## Synthetic Data Generators — `utils/generators.py`

`SyntheticFactory` generates spatial datasets with known regimes for benchmarking:

| Generator | Pattern |
|-----------|---------|
| `voronoi_sharp` | Voronoi cells with sharp boundaries |
| `linear_gradient` | Variance increases left-to-right |
| `concentric_donuts` | Rings around origin |
| `oblique_stripes` | Diagonal stripes |
| `sparse_islands` | Small high-variance hotspots |
| `spiral_volatility` | Archimedean spiral pattern |
| `checkerboard` | Checkerboard variance |
| `density_bias` | High variance in sparse regions |
| `nested_targets` | Nested circular clusters |
| `fractal_clouds` | Wave-interference pattern |
| `interlocking_moons` | Non-convex half-moons (challenging for GMM) |

Data is saved as `.npz` with keys: `x`, `y`, `z`, `labels`.

---

## Usage Example

```python
import pandas as pd
import numpy as np
from models.Clis.engine import Clis
from models.Clis.clis_forest import ClisForest
from models.Clis.metrics.evaluation import ClisEvaluator

# Load data: X has columns 'x', 'y'; y is the target variable (e.g., volatility)
X = pd.DataFrame({'x': ..., 'y': ...})
y = np.array([...])  # target variable

# Single tree
clis = Clis(
    loss_metric="pinball",
    complexity_penalty=0.01,
    lookahead_depth=0,
    min_samples_leaf=10
)
clis.fit(X, y)
labels_single = clis.predict(X)

# Ensemble (forest)
forest = ClisForest(
    n_estimators=10,
    n_clusters=5,  # number of final clusters
    bootstrap_sample_ratio=0.5,
    loss_metric="pinball",
    complexity_penalty=0.01
)
forest.fit(X, y)
labels_forest = forest.predict(X)

# Evaluate
evaluator = ClisEvaluator()
ari, nmi = evaluator.structural_scores(true_labels, labels_forest).values()
```

---

## Design Rationale

1. **Variance/regime focus**: Splits are chosen to reduce loss on \(z\), so clusters correspond to regions with different distributions (e.g., high vs. low variance).

2. **Geometric flexibility**: Multiple split types (axis, radial, oblique, elliptical) allow non-axis-aligned and non-convex boundaries.

3. **Statistical merging**: KS-based merging avoids over-partitioning when leaves have similar distributions.

4. **Scalability**: Forest uses leaf-embedding + MiniBatchKMeans instead of co-association matrices.

5. **Robustness**: Pinball loss is robust to outliers; bootstrap and ensemble reduce variance.

---

## File Structure

```
Clis/
├── engine.py           # Clis (single tree)
├── clis_forest.py      # ClisForest (ensemble)
├── split_strategies.py # Axis, Radial, Oblique, Elliptical splits
├── metrics/
│   └── evaluation.py   # ClisEvaluator
├── utils/
│   └── generators.py   # SyntheticFactory
└── DOCUMENTATION.md    # This file
```
