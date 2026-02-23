"""
Maximum Mean Discrepancy (MMD) for comparing multivariate distributions.

MMD is a kernel-based distance between probability distributions. It generalizes
to arbitrary dimensions and can compare joint distributions (unlike KS which is 1D).
"""

import numpy as np


def _rbf_kernel(X, Y, gamma):
    """Compute RBF (Gaussian) kernel matrix between X and Y."""
    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y)
    if X.shape[1] != Y.shape[1]:
        raise ValueError("X and Y must have same number of features")
    
    # Pairwise squared distances
    XX = np.sum(X ** 2, axis=1, keepdims=True)
    YY = np.sum(Y ** 2, axis=1, keepdims=True)
    XY = np.dot(X, Y.T)
    sq_dists = XX + YY.T - 2 * XY
    sq_dists = np.maximum(sq_dists, 0)  # numerical stability
    
    return np.exp(-gamma * sq_dists)


def mmd_squared(X, Y, gamma=None):
    """
    Unbiased estimator of MMD^2 between two samples.
    
    MMD^2 = E[k(x,x')] + E[k(y,y')] - 2*E[k(x,y)]
    
    Uses the unbiased U-statistic estimator (excludes diagonal).
    
    Parameters
    ----------
    X : array-like, shape (n_samples_X, n_features)
        First sample.
    Y : array-like, shape (n_samples_Y, n_features)
        Second sample.
    gamma : float, optional
        RBF kernel bandwidth (1 / (2*sigma^2)). If None, use median heuristic.
    
    Returns
    -------
    mmd2 : float
        Squared MMD statistic. Non-negative; 0 iff distributions are identical.
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    
    n_x, n_y = len(X), len(Y)
    if n_x < 2 or n_y < 2:
        return 0.0
    
    # Median heuristic for gamma if not provided
    if gamma is None:
        all_pts = np.vstack([X, Y])
        pairwise_sq = np.sum(all_pts ** 2, axis=1, keepdims=True)
        sq_dists = pairwise_sq + pairwise_sq.T - 2 * np.dot(all_pts, all_pts.T)
        sq_dists = np.maximum(sq_dists, 0)
        median_sq = np.median(sq_dists[sq_dists > 0]) if np.any(sq_dists > 0) else 1.0
        gamma = 1.0 / (2.0 * max(median_sq, 1e-8))
    
    K_xx = _rbf_kernel(X, X, gamma)
    K_yy = _rbf_kernel(Y, Y, gamma)
    K_xy = _rbf_kernel(X, Y, gamma)
    
    # Unbiased: exclude diagonal
    np.fill_diagonal(K_xx, 0)
    np.fill_diagonal(K_yy, 0)
    
    term_xx = np.sum(K_xx) / (n_x * (n_x - 1))
    term_yy = np.sum(K_yy) / (n_y * (n_y - 1))
    term_xy = 2 * np.mean(K_xy)
    
    mmd2 = term_xx + term_yy - term_xy
    return max(0.0, mmd2)


def mmd_null_pvalue(X, Y, gamma=None, n_permutations=100, random_state=None):
    """
    Permutation test for MMD: p-value under null that X and Y are from same distribution.
    
    High p-value -> distributions are similar -> cannot reject null.
    Low p-value -> distributions are different.
    
    Parameters
    ----------
    X, Y : array-like
        Samples to compare.
    gamma : float, optional
        RBF kernel bandwidth.
    n_permutations : int
        Number of permutation replicates.
    random_state : int, optional
        Random seed.
    
    Returns
    -------
    mmd2 : float
        Observed MMD^2 statistic.
    p_value : float
        Proportion of permuted MMD^2 >= observed. High p-value means similar distributions.
    """
    rng = np.random.default_rng(random_state)
    X = np.asarray(X)
    Y = np.asarray(Y)
    
    mmd_obs = mmd_squared(X, Y, gamma)
    
    # Pool and permute
    pooled = np.vstack([np.atleast_2d(X), np.atleast_2d(Y)])
    n_x = len(X)
    n_total = len(pooled)
    
    count_ge = 0
    for _ in range(n_permutations):
        perm = rng.permutation(n_total)
        X_perm = pooled[perm[:n_x]]
        Y_perm = pooled[perm[n_x:]]
        mmd_perm = mmd_squared(X_perm, Y_perm, gamma)
        if mmd_perm >= mmd_obs:
            count_ge += 1
    
    p_value = (1 + count_ge) / (1 + n_permutations)
    return mmd_obs, p_value
