# CLIS: Clustering with Loss-based Independence Splitting

---

## 1. Abstract and Motivation

**CLIS** (Clustering with Loss-based Independence Splitting) is a spatial clustering framework that discovers **regime-based partitions**: regions of a split space where a target variable (or joint target distribution) exhibits distinct statistical properties. Unlike traditional clustering methods (KMeans, Gaussian Mixture Models) that partition based on geometric proximity in feature space, CLIS partitions based on **statistical heterogeneity** of the target across the split space.

**Key insight**: In many applications (e.g., spatial volatility regimes, environmental risk zones, geographic heterogeneity in outcomes), the goal is not to cluster points by where they are, but by *how the outcome behaves* in different regions. CLIS explicitly optimizes for this by recursively splitting the space to minimize a loss on the target variable(s), then merging statistically indistinguishable leaves.

---

## 2. Problem Formulation

### 2.1 Data Structure

Let \(\mathcal{X} \subseteq \mathbb{R}^{d_s}\) denote the **split space** (e.g., spatial coordinates, covariates) and \(\mathcal{Y} \subseteq \mathbb{R}^{d_t}\) the **target space** (e.g., outcome, volatility, pollution level).

**Input**:
- \(X \in \mathbb{R}^{n \times d_s}\): split features (e.g., coordinates \(x, y\) or \(d_0, d_1, d_2\))
- \(Y \in \mathbb{R}^{n \times d_t}\): target variable(s); \(d_t = 1\) for univariate, \(d_t > 1\) for joint distribution

**Output**:
- Cluster labels \(\hat{c} \in \{0, 1, \ldots, K-1\}^n\) partitioning the \(n\) samples into \(K\) regimes

**Objective**: Partition the split space so that within each cluster, the target \(Y\) has a homogeneous distribution; across clusters, distributions differ (e.g., different variances, means, or full joint distributions).

### 2.2 Regime Interpretation

A **regime** is a region of \(\mathcal{X}\) where \(Y \mid X\) follows a distinct distribution. For example:
- **Variance regimes**: Same mean, different variance (e.g., low vs. high volatility)
- **Mean regimes**: Different means and possibly variances
- **Joint regimes**: Multivariate \(Y\) with different covariance structure (correlations, scale)

CLIS discovers these regimes by recursive binary splitting in \(\mathcal{X}\), guided by loss reduction on \(Y\).

---

## 3. The CLIS Algorithm

### 3.1 High-Level Overview

CLIS consists of two phases:

1. **Phase 1 — Recursive Splitting**: Build a decision tree over \(\mathcal{X}\) that minimizes a loss \(\mathcal{L}\) on \(Y\) at each split. Each leaf corresponds to a candidate regime.
2. **Phase 2 — Leaf Merging**: Merge leaves whose target distributions are statistically indistinguishable (via Kolmogorov-Smirnov for 1D, or Maximum Mean Discrepancy for multivariate \(Y\)).

### 3.2 Phase 1: Recursive Splitting

**Notation**:
- \(\mathcal{D} = \{(x_i, y_i)\}_{i=1}^n\): dataset
- \(\mathcal{L}(y)\): loss of target values \(y\) in a node
- \(\lambda\): complexity penalty (default 1.0)

**Split criterion**: At each node with data indices \(I\), choose a split \((s, \theta)\) that maximizes

\[
\text{Gain}(s, \theta) = \mathcal{L}(y_I) - \bigl[\mathcal{L}(y_{I_L}) + \mathcal{L}(y_{I_R})\bigr] - \lambda \log |I|
\]

where \(I_L\) and \(I_R\) are the left and right child indices induced by split \((s, \theta)\), and \(\lambda \log |I|\) penalizes splits at large nodes to avoid overfitting.

**Stopping conditions**:
- \(|I| < 2 \cdot m_{\min}\) (minimum leaf size): We need at least \(m_{\min}\) points in each child to compute a meaningful loss and avoid tiny, unstable leaves.
- No split achieves \(\text{Gain} \geq \tau_{\text{gain}}\) (effective threshold): No split improves the objective enough.
- No valid split passes the minimum leaf size constraint: Every proposal that reduces loss creates a child with fewer than \(m_{\min}\) points, so we reject it.

**Effective threshold**: \(\tau_{\text{eff}} = \tau_{\text{gain}} \cdot |\mathcal{L}(y_{\text{root}})|\) so the threshold scales with the initial loss. A raw threshold of 0.001 would be meaningless if the root loss is 10,000. Scaling by the root loss keeps the criterion meaningful across datasets.

**BFS order**: We process nodes in breadth-first order (queue). This ensures we fully expand the root before going deep, which tends to find the most impactful splits first.

### 3.3 Lookahead (Optional)

For marginal splits (gain not clearly dominant), CLIS can perform **lookahead**: recursively evaluate the best possible sub-splits in the left and right children and add their potential gains to the path gain. This helps avoid myopic splits.

- **Adaptive proposals**: 20 split proposals at the root, 5 at deeper nodes (reduces computation).
- **Early exit**: If immediate gain exceeds \(5 \cdot \tau_{\text{gain}}\), skip lookahead.
- **Min-depth warm start**: Below minimum depth, force a small positive gain to prevent premature stopping.

### 3.4 Phase 2: Leaf Merging

After splitting, leaves may have statistically similar target distributions (e.g., same variance). Merging reduces over-partitioning.

**Single target** (\(d_t = 1\)): **Kolmogorov-Smirnov (KS) two-sample test**
- \(H_0\): The two samples come from the same distribution.
- Merge leaves \(A\) and \(B\) if \(p\text{-value} > \tau_{\text{merge}}\).

**Joint target** (\(d_t > 1\)): **Maximum Mean Discrepancy (MMD)**
- **Threshold mode** (fast): Merge if \(\text{MMD}^2(y_A, y_B) < \tau_{\text{MMD}}\).
- **Permutation mode** (slower): Merge if permutation \(p\)-value \(> \tau_{\text{merge}}\).

**Union-Find**: Merging is implemented via Union-Find; all leaves in the same equivalence class receive the same final cluster label.

---

## 3.5 Deep Dive: How the Algorithm Works and Why

This section explains the mechanics and rationale behind each design choice.

### Why Loss-Based Splitting (Instead of Geometry)?

**KMeans and GMM** partition by **where** points are: they cluster in \((x, y, z)\) space by proximity. If your goal is to find regions where the *outcome behaves differently* (e.g., high vs. low variance), that geometry may not align with regime boundaries. Two regions can be spatially close but have very different outcome distributions.

**CLIS** uses the *target* to guide splits. At each node, we ask: "Is there a cut in the split space that separates points with different target behavior?" The loss measures how "mixed" the target is in a node. A good split reduces the total loss: the left and right children each have a more homogeneous target distribution than the parent. So we're explicitly optimizing for regime separation.

**Concrete example**: Suppose you have (x, y) coordinates and a volatility measure z. Region A has low variance (z stable), Region B has high variance (z spiky). They might overlap in (x, y). KMeans would cluster by (x, y, z) proximity—points with similar z might get grouped, but the boundary would be driven by all three. CLIS splits (x, y) to minimize loss on z, so regions with different z distributions get separated even if they're spatially interleaved.

### Why the Gain Formula?

\[
\text{Gain} = \mathcal{L}(\text{parent}) - \bigl[\mathcal{L}(\text{left}) + \mathcal{L}(\text{right})\bigr] - \lambda \log |I|
\]

- **\(\mathcal{L}(\text{parent}) - [\mathcal{L}(\text{left}) + \mathcal{L}(\text{right})]\)**: This is the *raw* reduction in loss. If the split is good, the children are more homogeneous than the parent, so their combined loss is lower. We want to maximize this reduction.

- **\(-\lambda \log |I|\)**: The *complexity penalty*. Without it, we could always reduce loss by splitting further—eventually each leaf would have one point and zero loss. The penalty grows with node size: splitting a large node costs more than splitting a small one. This discourages splits that only marginally improve the loss and prevents overfitting. The \(\log\) makes the penalty scale sublinearly so we don't over-penalize large nodes.

### Why an Effective Threshold That Scales?

The raw `gain_threshold` (e.g., 0.001) is tiny—it would be meaningless if the root loss is 10,000. So we use \(\tau_{\text{eff}} = \tau_{\text{gain}} \cdot |\mathcal{L}(y_{\text{root}})|\). If the root loss is large, we require a proportionally larger gain to split. This keeps the stopping criterion meaningful across datasets with different scales.

### Why Multiple Split Strategies?

**Axis**: Simple, fast, works for axis-aligned boundaries. But if the true regime boundary is a diagonal line or a circle, axis splits need many cuts to approximate it.

**Radial**: Cuts by distance from a center. One split can separate a circular region from the rest. Good for rings, concentric shells.

**Oblique**: Cuts by a linear combination of coordinates. One split can capture a diagonal line. Good for stripes, oblique boundaries.

**Elliptical**: Cuts by an ellipse/ellipsoid. Handles rotated or elongated regions that aren't axis-aligned or circular.

By using all four, we increase the chance of finding a split that matches the true boundary shape. Each strategy proposes random candidates; we pick the best across all of them. The randomization (percentiles, jitter) avoids always splitting at the median or centroid, which can be suboptimal.

### Why Lookahead?

Sometimes the *immediate* gain from a split is small, but that split enables *future* splits that are very good. A myopic algorithm might reject it. Lookahead: for marginal splits (gain not clearly dominant), we recursively evaluate the best possible sub-splits in the left and right children and add their gains to the path gain. This helps us choose splits that set up the tree for better downstream structure.

**Example**: Splitting at depth 0 might give gain 0.1, but each child could then be split with gain 5. The path gain would be 0.1 + 5 + 5 = 10.1, making this split attractive. Without lookahead, we might prefer a different split with immediate gain 2 that leads to worse structure.

**Cost**: Lookahead is expensive. So we use it only when the immediate gain is marginal (below \(5 \cdot \tau_{\text{gain}}\)), and we use fewer proposals at deeper nodes (5 instead of 20) to reduce the search.

### Why Merging?

The splitting phase is greedy: it keeps splitting as long as the gain exceeds the threshold. This can produce:

1. **Over-partitioning**: Two leaves might have nearly identical target distributions (e.g., same variance). They're separate only because the tree found some small gain to split them. Statistically, they're the same regime.

2. **Fragmentation**: The tree might split on noise in small regions, creating many tiny leaves.

**Merging** fixes this by comparing leaves pairwise. If two leaves have distributions we can't distinguish (high KS p-value or low MMD), we merge them. The result is a coarser, more interpretable partition that still respects the statistical structure.

**Why KS for 1D?** The Kolmogorov-Smirnov test compares two samples by the maximum difference between their empirical CDFs. It's non-parametric and works for any distribution. High p-value means we can't reject "same distribution" → merge.

**Why MMD for multi-D?** KS is for univariate data. For joint distributions (e.g., 2D target), we need a multivariate test. MMD is a kernel-based distance between distributions; it generalizes to any dimension and can capture correlations.

### Why a Union-Find for Merging?

We compare all pairs of leaves. If A and B are similar, we merge them. If B and C are similar, we merge them. Then A, B, C should all be in the same cluster. Union-Find keeps track of these equivalence classes: when we merge A and B, we union their sets; at the end, each leaf's cluster ID is its representative in the Union-Find structure.

### Walkthrough: One Split Step

1. **Node**: We have indices \(I\) (e.g., 500 points) in the current node.

2. **Parent loss**: \(\mathcal{L}(y_I) = 1200\) (e.g., pinball loss over the node's target values).

3. **Proposals**: For each strategy, we sample several splits. E.g., axis: "x < 3.2"; radial: "distance from (1,2) < 4"; oblique: "0.7x + 0.7y < 5"; etc.

4. **For each proposal**: Apply the split → get \(I_L\) and \(I_R\). Compute \(\mathcal{L}(y_{I_L})\) and \(\mathcal{L}(y_{I_R})\). Say \(\mathcal{L}(y_{I_L}) = 200\), \(\mathcal{L}(y_{I_R}) = 300\). Raw reduction = 1200 - 500 = 700. Penalty = \(\lambda \log 500 \approx 6.2\). Gain = 700 - 6.2 = 693.8.

5. **Best**: If 693.8 is the best across all proposals and exceeds \(\tau_{\text{eff}}\), we store this split and enqueue the two children.

6. **If no best**: We make this node a leaf and record its target data for the merge phase.

### End-to-End Execution Trace

**Input**: \(X\) (e.g., 1000 × 2 coordinates), \(Y\) (e.g., 1000 × 1 outcome).

1. **Setup**: Build internal DataFrame with columns [x, y, _z0]. Resolve split_cols = [x, y]. Create strategy map for these columns. Initialize queue = [(node_0, all_indices)].

2. **Split phase (BFS)**:
   - Pop (node_0, indices). |indices| = 1000 ≥ 2×10, so try to split.
   - Call _evaluate_lookahead: for axis, radial, oblique, elliptical, propose 20 splits each. For each, compute gain. Best gain = 450, exceeds threshold. Store tree_[0] = (("radial", params), left_id=1, right_id=2). Enqueue (1, left_indices), (2, right_indices).
   - Pop (1, left_indices). |left_indices| = 600. Best split found, gain = 200. Store tree_[1], enqueue children.
   - Continue until nodes are too small or no split helps. Suppose we end up with leaves 5, 6, 7, 8, 9, 10.

3. **Merge phase**: leaf_data_map = {5: y_5, 6: y_6, ...}. For each pair (5,6), (5,7), ..., compute KS or MMD. Suppose (5,6) have p-value 0.8 → merge. (7,8) have p-value 0.9 → merge. Union-Find: 5↔6, 7↔8. merge_map_ = {5:5, 6:5, 7:7, 8:7, 9:9, 10:10}. Final clusters: {5,6}, {7,8}, {9}, {10}.

4. **Predict**: For a new point (2.1, 3.4), route down: node 0 → radial says inside → node 1 → axis says left → node 5 (leaf). merge_map_[5] = 5. Assign label 5.

### Why Two Phases (Split Then Merge)?

**Splitting alone** would give us as many leaves as the gain threshold allows. That can be too many: the tree might split on small, noisy differences, creating fragments that are statistically the same regime. We'd have 20 leaves when there are really 4 regimes.

**Merging** corrects this. After splitting, we compare leaves pairwise. If two leaves have indistinguishable target distributions (high KS p-value or low MMD), we merge them. The result: we keep the spatial resolution from the tree (the splits found meaningful boundaries) but collapse leaves that don't differ statistically. So we get 4–6 merged clusters instead of 20 raw leaves.

**Order matters**: We merge *after* splitting because we need the leaves first. The split phase is greedy and doesn't know the final cluster count; the merge phase uses distributional tests to decide which leaves are redundant.

---

## 4. Split Strategies

Splits are defined by a **strategy** \(s\) and **parameters** \(\theta\). Each strategy has:
- `propose(data)`: Sample candidate \(\theta\) (randomized over data-dependent ranges).
- `apply(data, θ)`: Return boolean mask for left child (points satisfying the split condition).

### 4.1 Axis-Aligned Split

**Condition**: \(x_j < v\) for some dimension \(j\) and threshold \(v\).

**Parameters**: \(\theta = (j, v)\), with \(j\) chosen uniformly and \(v \sim \text{Uniform}(q_{20}, q_{80})\) (20th–80th percentile of \(x_j\)).

**How it works**: Pick a random dimension (e.g., x or y in 2D). Pick a threshold between the 20th and 80th percentiles of that dimension. Left = points below the threshold. We avoid the 0–20% and 80–100% range so we don't create trivial splits (e.g., 2 points on one side).

**Why percentiles?** The median would often split 50-50, which might not align with regime boundaries. Sampling in [q20, q80] explores off-center splits that can better separate regimes.

**Use case**: Axis-aligned boundaries, simple box partitions.

### 4.2 Radial Split

**Condition**: \(\|x - c\|_2 < r\) for center \(c\) and radius \(r\).

**Parameters**: \(c\) is a sampled point plus jitter (\(\pm 10\%\) of range); \(r \sim \text{Uniform}(q_{10}, q_{85})\) of distances from \(c\).

**How it works**: Pick a random data point as the center, then add small random jitter so the circle isn't exactly centered on a sample. Compute distances from all points to this center. Pick a radius between the 10th and 85th percentiles of those distances. Left = points inside the circle.

**Why jitter?** If we always use a data point as center, we might bias toward that point's neighborhood. Jitter breaks symmetry and explores different centers.

**Why q10–q85 for radius?** Too small (e.g., q5) gives a tiny inner region; too large (e.g., q95) gives a huge inner region. The middle range yields balanced splits.

**Use case**: Circular/ring-shaped regimes, radial symmetry.

### 4.3 Oblique Split

**Condition**: \(\langle w, x \rangle < b\) for unit vector \(w\) and intercept \(b\).

**Parameters**: \(w\) is a random unit vector (uniform on sphere); \(b \sim \text{Uniform}(q_{15}, q_{85})\) of projections \(\langle w, x \rangle\).

**How it works**: Sample a random direction \(w\) (normalize a Gaussian vector to get uniform on the sphere). Project all points onto this direction: \(p_i = \langle w, x_i \rangle\). Pick an intercept \(b\) between the 15th and 85th percentiles of these projections. Left = points with projection below \(b\). This defines a hyperplane perpendicular to \(w\).

**Why random direction?** Axis splits only consider coordinate axes. Oblique splits can capture diagonal boundaries (e.g., "x + y < 5") with a single cut. Random directions explore the space of possible hyperplanes.

**Use case**: Diagonal stripes, anisotropic patterns, linear boundaries.

### 4.4 Elliptical Split

**Condition**: \(\|R^\top (x - c) \oslash a\|_2^2 < 1\), where \(R\) is a rotation matrix, \(a\) are semi-axes, and \(\oslash\) is element-wise division.

**Parameters**: \(c\) sampled with jitter; \(a\) random in \([0.2, 0.9] \times \max_i \|x_i - c\|\); \(R\) from QR of random matrix (or 2D rotation by random angle).

**How it works**: Center the data at \(c\), rotate by \(R\), scale by semi-axes \(a\). The ellipse equation \((x'/a_x)^2 + (y'/a_y)^2 < 1\) defines the interior. Left = points inside the ellipse. The rotation allows the ellipse to be oriented in any direction; different \(a\) values allow elongation.

**Why ellipse instead of circle?** Radial splits are circles. Elliptical splits handle elongated or rotated regions (e.g., a diagonal cigar shape) that a circle would poorly approximate.

**Use case**: Elliptical/ellipsoidal regions, elongated clusters.

**Generalization**: All strategies operate on an arbitrary subset of dimensions (`split_cols`), supporting 2D, 3D, or higher-dimensional split spaces.

---

## 5. Loss Functions

The loss \(\mathcal{L}(y)\) measures the *inhomogeneity* of the target in a node. Splits that reduce total loss are preferred. The idea: a node with a wide, spread-out target distribution has high loss; a node with a tight, homogeneous distribution has low loss. A good split sends different parts of the distribution to different children, so each child has lower loss than the parent.

### 5.1 Univariate Target (\(d_t = 1\))

**MSE (Mean Squared Error)**:
\[
\mathcal{L}_{\text{MSE}}(y) = \sum_{i=1}^{n} (y_i - \bar{y})^2
\]
**What it measures**: Sum of squared deviations from the mean. This is exactly \(n\) times the variance. So minimizing MSE is equivalent to variance reduction: we want splits that create children with lower variance than the parent.

**When to use**: Simple and fast. Good when you care mainly about variance (e.g., volatility regimes). Sensitive to outliers—a few extreme values can dominate the loss.

**NLL (Gaussian Negative Log-Likelihood)**:
\[
\mathcal{L}_{\text{NLL}}(y) = \frac{n}{2} \log(\hat{\sigma}^2) + \frac{n}{2}, \quad \hat{\sigma}^2 = \frac{1}{n}\sum_i (y_i - \bar{y})^2
\]
**What it measures**: How well the data fits a Gaussian with the sample mean and variance. The NLL is the negative log of the Gaussian likelihood; lower NLL = better fit. It penalizes both high variance (the \(\log \sigma^2\) term) and misfit (implicit in the likelihood).

**When to use**: When you assume the target is roughly Gaussian in each regime. Slightly different from MSE in how it weighs the variance term.

**Pinball (Quantile Loss)**:
\[
\mathcal{L}_{\text{pinball}}(y) = \sum_{q \in \mathcal{Q}} \sum_{i=1}^{n} \rho_q(y_i - \hat{y}_q), \quad \rho_q(u) = \max(qu, (q-1)u)
\]
where \(\mathcal{Q} = \{0.01, 0.05, 0.1, \ldots, 0.95, 0.99\}\) and \(\hat{y}_q\) is the empirical \(q\)-quantile.

**What it measures**: For each quantile \(q\), we predict the \(q\)-quantile and penalize residuals asymmetrically. If \(y_i > \hat{y}_q\), we penalize by \(q \cdot (y_i - \hat{y}_q)\); if \(y_i < \hat{y}_q\), we penalize by \((q-1) \cdot (y_i - \hat{y}_q)\). Sum over many quantiles (0.01 to 0.99) to capture the full distribution—not just mean and variance, but the shape of the tails, skew, etc.

**Why pinball?** (1) **Robust**: Outliers affect only the extreme quantiles; the bulk of the loss is from the middle quantiles. (2) **Distribution-aware**: Two distributions with the same mean and variance can have different quantiles (e.g., different skew). Pinball distinguishes them. (3) **No distributional assumption**: Unlike NLL, we don't assume Gaussianity.

**When to use**: Default choice when the target may have outliers or non-Gaussian shape. Slightly more expensive than MSE/NLL because we compute many quantiles.

### 5.2 Multivariate Target (\(d_t > 1\))

**Multivariate Gaussian NLL** (always used for \(d_t > 1\)):
\[
\mathcal{L}_{\text{NLL}}(Y) = \frac{n}{2} \log \det(\hat{\Sigma}) + \frac{n \cdot d_t}{2}
\]
where \(\hat{\Sigma}\) is the sample covariance of \(Y\).

**What it measures**: The determinant \(\det(\hat{\Sigma})\) is the "generalized variance"—it captures the overall spread and correlation structure. If two dimensions are highly correlated, the covariance matrix is "narrow" in that direction and the determinant is smaller. The loss penalizes large generalized variance: we want tight, coherent clusters in the joint space.

**Why not MSE or pinball for multi-D?** 
- **MSE**: Summing variances over dimensions treats each dimension independently. It ignores correlations. Two clusters could have the same marginal variances but different correlations; MSE wouldn't distinguish them.
- **Pinball**: Summing pinball over dimensions also ignores correlations. Each dimension is scored separately. The joint distribution (e.g., "when d3 is high, d4 tends to be low") is not captured.

**Why multivariate NLL?** The covariance matrix \(\hat{\Sigma}\) encodes both variance per dimension and correlations between dimensions. The determinant is a single scalar that summarizes this. Minimizing it encourages splits that create children with tighter, more coherent joint distributions.

**Design choice**: For \(d_t > 1\), `loss_metric` is ignored and multivariate NLL is used, because MSE and pinball do not properly capture joint distributions.

---

## 6. Maximum Mean Discrepancy (MMD)

For merging leaves with multivariate targets, we use MMD, a kernel-based distance between distributions. KS only works for 1D; MMD generalizes to any dimension and can compare full joint distributions.

### 6.1 Definition and Intuition

Let \(k: \mathcal{Y} \times \mathcal{Y} \to \mathbb{R}\) be a kernel. The squared MMD between distributions \(P\) and \(Q\) is
\[
\text{MMD}^2(P, Q) = \mathbb{E}_{x,x' \sim P}[k(x,x')] + \mathbb{E}_{y,y' \sim Q}[k(y,y')] - 2\mathbb{E}_{x \sim P, y \sim Q}[k(x,y)]
\]

**Intuition**: The first term is the average similarity of points within \(P\); the second, within \(Q\); the third, between \(P\) and \(Q\). If \(P = Q\), within- and cross-similarities match, so MMD² = 0. If \(P\) and \(Q\) differ, points from \(P\) are less similar to points from \(Q\) than to each other, so the cross term is smaller and MMD² > 0.

**Property**: \(\text{MMD}^2(P, Q) = 0\) iff \(P = Q\) (for characteristic kernels such as the RBF kernel).

### 6.2 RBF Kernel and Median Heuristic

We use the RBF (Gaussian) kernel:
\[
k(x, y) = \exp\bigl(-\gamma \|x - y\|^2\bigr)
\]

**What the kernel does**: \(k(x,y)\) is close to 1 when \(x\) and \(y\) are near each other, and decays to 0 as they move apart. Large \(\gamma\) = narrow kernel = only very close points are "similar"; small \(\gamma\) = wide kernel = points can be far and still "similar."

**Bandwidth \(\gamma\)**: If not specified, we use the **median heuristic**: \(\gamma = 1 / (2 \cdot \text{median}(\|x_i - x_j\|^2))\) over distinct pairs. The median squared distance gives a typical scale; this sets \(\gamma\) so the kernel adapts to the data. No manual tuning needed.

### 6.3 Unbiased Estimator

For samples \(X = \{x_1, \ldots, x_{n_x}\}\) and \(Y = \{y_1, \ldots, y_{n_y}\}\), the unbiased U-statistic estimator (excluding diagonal terms) is:
\[
\widehat{\text{MMD}}^2 = \frac{1}{n_x(n_x-1)}\sum_{i \neq j} k(x_i, x_j) + \frac{1}{n_y(n_y-1)}\sum_{i \neq j} k(y_i, y_j) - \frac{2}{n_x n_y}\sum_{i,j} k(x_i, y_j)
\]

**Why exclude diagonal?** The diagonal terms \(k(x_i, x_i) = 1\) don't carry information about the distribution. The U-statistic form gives an unbiased estimate of the population MMD².

### 6.4 Permutation Test and Threshold Mode

**Permutation test**: Under \(H_0: P = Q\), we pool the samples, randomly reassign labels, and compute MMD² for many permutations. The p-value is the proportion of permuted MMD² values ≥ the observed value. High p-value → we can't reject "same distribution" → merge.

**Threshold mode (faster)**: Instead of permutations, we merge when MMD² < \(\tau_{\text{MMD}}\). No permutations needed; typical values are 0.05–0.2 depending on data scale.

---

## 7. ClisForest: Ensemble Extension

### 7.1 Why an Ensemble?

A single CLIS tree can be sensitive to the specific splits it finds. Different random proposals can lead to different trees. An ensemble of trees, each fit on a bootstrap sample, reduces this variance: samples that truly belong to the same regime will tend to land in similar leaves across trees, while noise is averaged out.

### 7.2 Training

- Train \(T\) CLIS trees on **bootstrap samples** (e.g., 50% of data per tree, sampled with replacement).
- Each tree is fit independently with the same hyperparameters and `split_cols`.
- Bootstrap sampling gives each tree a slightly different view of the data, so the trees diversify.

### 7.3 Prediction: Scalable Consensus

**The problem**: A standard ensemble approach is to build a co-association matrix: entry \((i,j)\) = fraction of trees where samples \(i\) and \(j\) land in the same leaf. Then cluster this matrix (e.g., spectral clustering). But the matrix is \(N \times N\)—prohibitive for large \(N\).

**Our approach**: Instead of the full matrix, we use a **leaf embedding**:

1. **Leaf embedding**: For each sample \(i\), collect the leaf ID from each tree → vector \(\ell_i \in \mathbb{Z}^T\). So each sample is a \(T\)-dimensional vector of leaf IDs.
2. **Consensus clustering**: Run **MiniBatchKMeans** on the matrix \([\ell_1, \ldots, \ell_N]^\top\) with \(K\) clusters.
3. **Output**: Cluster labels from KMeans.

**Why this works**: Samples from the same regime tend to follow similar paths down each tree and land in the same (or nearby) leaves. So their leaf-ID vectors are similar. KMeans clusters these vectors: samples with similar leaf signatures get the same label. The leaf IDs act as a categorical embedding—we're clustering in "leaf space" instead of building the full co-association matrix.

**Complexity**: \(O(N \cdot T)\) for leaf assignment plus \(O(N)\) for MiniBatchKMeans. We avoid \(O(N^2)\) storage and \(O(N^2)\) or \(O(N^3)\) clustering.

**Note**: You must specify \(K\) (the number of clusters) for the forest. The single tree doesn't estimate \(K\); it produces as many merged leaves as the data supports. The forest uses KMeans with a user-specified \(K\) to get a fixed number of clusters.

---

## 8. Implementation Details

### 8.1 Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `split_cols` | None (all) | Columns of \(X\) used for splitting |
| `min_samples_leaf` | 10 | Minimum samples per leaf |
| `gain_threshold` | 0.001 | Minimum gain to accept a split |
| `loss_metric` | `"pinball"` | For 1D: `"mse"`, `"nll"`, `"pinball"` |
| `complexity_penalty` | 1.0 | \(\lambda\) in split criterion |
| `lookahead_depth` | 2 | Depth of lookahead for marginal splits |
| `merge_threshold` | 0.005 | KS/MMD permutation \(p\)-value threshold |
| `merge_metric` | `"auto"` | `"ks"` (1D), `"mmd"` (multi-D), or `"auto"` |
| `merge_use_permutation` | False | If False, use MMD² threshold (faster) |
| `merge_mmd_threshold` | 0.1 | MMD² threshold when not using permutation |
| `strategies` | axis, radial, oblique, elliptical | Split strategies to use |

### 8.2 Data Flow

1. **Fit**: \(X, Y\) → internal DataFrame with split columns + target columns (`_z0`, `_z1`, …).
2. **Split phase**: BFS over nodes; for each node, evaluate proposals from all strategies, pick best gain, enqueue children.
3. **Merge phase**: For each pair of leaves, compute KS or MMD; merge if similar; apply Union-Find.
4. **Predict**: Route each point down the tree to a leaf; map leaf ID to merged cluster ID.

### 8.3 Scalability Considerations

- **Splitting**: \(O(\text{proposals} \times \text{strategies} \times n_{\text{node}})\) per node; adaptive proposals reduce cost at depth.
- **Merging**: \(O(L^2)\) leaf pairs; MMD threshold mode avoids permutation cost.
- **Forest**: Bootstrap and MiniBatchKMeans keep memory and time manageable for large \(N\).

---

## 9. Comparison to Alternative Methods

| Method | Partition criterion | Handles non-convex? | Joint distribution? | Variance-only regimes? |
|--------|---------------------|----------------------|---------------------|-------------------------|
| **KMeans** | Geometric (centroids) | No (convex cells) | Via concatenation | Weak |
| **GMM** | Probabilistic (ellipsoids) | No (ellipses) | Via concatenation | Weak (overlaps in mean) |
| **CLIS** | Loss on target | Yes (recursive splits) | Yes (MMD, multivariate NLL) | Yes (explicit) |

**When CLIS excels**:
- Non-convex regime boundaries (spirals, rings, moons).
- Variance-only or covariance-different regimes (same mean, different scale/correlation).
- Density–variance mismatch (e.g., dense low-variance core vs. sparse high-variance periphery).
- Sharp, non-elliptical boundaries (checkerboards, oblique stripes).

---

## 10. Evaluation Metrics

The `ClisEvaluator` class provides:

- **ARI** (Adjusted Rand Index): Agreement with ground truth, corrected for chance.
- **NMI** (Normalized Mutual Information): Information-theoretic overlap.
- **Variance contrast**: Variance of cluster variances (separation of regimes).
- **Boundary leakage**: Proportion of points misassigned relative to true boundaries.
- **Spatial hinge loss**: Distance-based penalty for misclassified points.
- **Boundary variance starkness**: Mean absolute difference in variance across discovered boundaries.
- **Distribution continuity**: Mean KS statistic between cluster pairs.

---

## 11. Synthetic Data Generators

### 11.1 2D Generators (`utils/generators.py`)

| Generator | Regime structure | Challenge |
|-----------|------------------|-----------|
| `voronoi_sharp` | Voronoi cells | Baseline |
| `linear_gradient` | Variance gradient in \(x\) | Continuous change |
| `concentric_donuts` | Rings | Non-convex |
| `oblique_stripes` | Diagonal stripes | Oblique boundaries |
| `sparse_islands` | Hotspots | Sparse high-variance |
| `spiral_volatility` | Spiral band | Non-convex spiral |
| `checkerboard` | Grid | Sharp boundaries |
| `density_bias` | Core vs. outer | Density–variance mismatch |
| `nested_targets` | Nested circles | Nested non-convex |
| `fractal_clouds` | Wave interference | Complex boundaries |
| `interlocking_moons` | Half-moons | Classic GMM failure |

### 11.2 5D Generators (`experiments/run_experiment_5d.py`)

- **voronoi**: 3D Voronoi in split space; distinct bivariate Gaussians in target.
- **spiral_volatility**: Spiral band in 3D; inside vs. outside differ in (d3, d4).
- **density_bias**: Sphere vs. outer; different joint distributions.
- **concentric_shells**: Three 3D shells; GMM fits ellipsoids, not shells.
- **variance_only**: Same mean in (d3, d4), different covariances.
- **checkerboard**: 3D grid with alternating regimes.

---

## 12. File Structure

```
Clis/
├── engine.py              # Clis (single tree)
├── clis_forest.py         # ClisForest (ensemble)
├── split_strategies.py    # Axis, Radial, Oblique, Elliptical
├── metrics/
│   ├── evaluation.py      # ClisEvaluator
│   └── mmd.py             # MMD implementation
├── utils/
│   └── generators.py      # SyntheticFactory (2D)
└── DOCUMENTATION.md       # This file
```

---

## 13. Usage Examples

### 13.1 2D Split, Univariate Target

```python
from models.Clis.engine import Clis

X = pd.DataFrame({'x': x_coords, 'y': y_coords})
y = np.array(z_values)  # 1D target

clis = Clis(loss_metric="pinball", complexity_penalty=0.01)
clis.fit(X, y)
labels = clis.predict(X)
```

### 13.2 5D: Split on 3, Joint Target on 2

```python
X = pd.DataFrame({'d0': ..., 'd1': ..., 'd2': ...})  # split space
y = np.column_stack([d3, d4])  # joint target, shape (n, 2)

clis = Clis(
    split_cols=['d0', 'd1', 'd2'],
    loss_metric="nll",  # multi-D uses NLL
    merge_metric="mmd",
    merge_use_permutation=False,
    merge_mmd_threshold=0.15
)
clis.fit(X, y)
labels = clis.predict(X)
```

### 13.3 Forest

```python
from models.Clis.clis_forest import ClisForest

forest = ClisForest(
    n_estimators=10,
    n_clusters=5,
    split_cols=['d0', 'd1', 'd2'],
    **clis_params
)
forest.fit(X, y)
labels = forest.predict(X)
```

