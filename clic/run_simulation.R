# ==========================================
# 0. SETUP
# ==========================================
library(ggplot2)
library(mclust)
library(mcclust)
# library(mcclust.ext) # Use if available, else minbinder is used below
library(Rcpp)
library(RcppArmadillo)
library(gridExtra)
library(MASS) # For multivariate normal (Anisotropic)

# Load the C++ functions
sourceCpp("rcppfuncts/sampling.cpp")
sourceCpp("rcppfuncts/postprocessing.cpp")

set.seed(42)

# ==========================================
# 1. DATA GENERATION FUNCTIONS (10 SCENARIOS)
# ==========================================

# 1. Baseline Blobs
gen_baseline_blobs <- function(n=2000) {
  X <- numeric(0); Y <- numeric(0); Z <- numeric(0); L <- numeric(0)
  centers <- matrix(c(-5,-5, 5,5, 5,-5, -5,5, 0,0), ncol=2, byrow=TRUE)
  
  pts_per <- n %/% 5
  for (i in 1:5) {
    X <- c(X, rnorm(pts_per, mean=centers[i,1], sd=1.5))
    Y <- c(Y, rnorm(pts_per, mean=centers[i,2], sd=1.5))
    Z <- c(Z, rnorm(pts_per, mean=10*(i-1), sd=5.0))
    L <- c(L, rep(i-1, pts_per))
  }
  return(data.frame(x=X, y=Y, z=Z, true_label=L))
}

# 2. Sharp Variance (Voronoi)
gen_sharp_voronoi <- function(n=2000) {
  x <- runif(n, -15, 15)
  y <- runif(n, -15, 15)
  points <- cbind(x, y)
  seeds <- matrix(c(-8,-8, 5,-5, 0,8, 10,10, -10,5), ncol=2, byrow=TRUE)
  
  # Voronoi assignment via distance matrix
  dists <- as.matrix(dist(rbind(points, seeds)))
  dists <- dists[1:n, (n+1):(n+5)]
  labels <- apply(dists, 1, which.min) - 1 # 0-indexed for consistency
  
  z_sigmas <- c(0.1, 5.0, 20.0, 100.0, 200.0)
  z <- numeric(n)
  
  for (i in 0:4) {
    mask <- labels == i
    if (sum(mask) > 0) {
      z[mask] <- rnorm(sum(mask), mean=50, sd=z_sigmas[i+1])
    }
  }
  return(data.frame(x=x, y=y, z=z, true_label=labels))
}

# 3. Extreme Density Imbalance
gen_density_imbalance <- function(n=2000) {
  X <- numeric(0); Y <- numeric(0); Z <- numeric(0); L <- numeric(0)
  props <- c(0.02, 0.03, 0.05, 0.1, 0.8)
  centers <- matrix(c(-8,-8, 8,8, -8,8, 8,-8, 0,0), ncol=2, byrow=TRUE)
  
  for (i in 1:5) {
    n_c <- floor(n * props[i])
    X <- c(X, rnorm(n_c, mean=centers[i,1], sd=1.0))
    Y <- c(Y, rnorm(n_c, mean=centers[i,2], sd=1.0))
    Z <- c(Z, rnorm(n_c, mean=50, sd=10.0))
    L <- c(L, rep(i-1, n_c))
  }
  return(data.frame(x=X, y=Y, z=Z, true_label=L))
}

# 4. Variance Imbalance
gen_variance_imbalance <- function(n=2000) {
  X <- numeric(0); Y <- numeric(0); Z <- numeric(0); L <- numeric(0)
  centers <- matrix(c(-6,0, -3,0, 0,0, 3,0, 6,0), ncol=2, byrow=TRUE)
  sigmas <- c(0.1, 1.0, 10.0, 50.0, 100.0)
  
  pts_per <- n %/% 5
  for (i in 1:5) {
    X <- c(X, rnorm(pts_per, mean=centers[i,1], sd=0.8))
    Y <- c(Y, rnorm(pts_per, mean=centers[i,2], sd=0.8))
    Z <- c(Z, rnorm(pts_per, mean=50, sd=sigmas[i]))
    L <- c(L, rep(i-1, pts_per))
  }
  return(data.frame(x=X, y=Y, z=Z, true_label=L))
}

# 5. Anisotropic Clusters
gen_anisotropic <- function(n=2000) {
  X <- numeric(0); Y <- numeric(0); Z <- numeric(0); L <- numeric(0)
  means <- list(c(0,0), c(5,5), c(-5,-5))
  covs <- list(matrix(c(5,4,4,5),2), matrix(c(1,-0.8,-0.8,1),2), matrix(c(2,0,0,10),2))
  
  pts_per <- n %/% 3
  for (i in 1:3) {
    pts <- mvrnorm(pts_per, means[[i]], covs[[i]])
    X <- c(X, pts[,1])
    Y <- c(Y, pts[,2])
    Z <- c(Z, rnorm(pts_per, mean=10*(i-1), sd=5.0))
    L <- c(L, rep(i-1, pts_per))
  }
  return(data.frame(x=X, y=Y, z=Z, true_label=L))
}

# 6. Radial Heteroscedasticity
gen_radial_hetero <- function(n=2000) {
  x <- runif(n, -10, 10)
  y <- runif(n, -10, 10)
  r <- sqrt(x^2 + y^2)
  
  labels <- rep(0, n)
  labels[r >= 3 & r < 6] <- 1
  labels[r >= 6] <- 2
  
  z <- rnorm(n, mean=50, sd=(1 + r^2))
  return(data.frame(x=x, y=y, z=z, true_label=labels))
}

# 7. Linear Gradient Variance
gen_linear_gradient <- function(n=2000) {
  x <- runif(n, 0, 20)
  y <- runif(n, -5, 5)
  
  labels <- rep(0, n)
  labels[x >= 7 & x < 14] <- 1
  labels[x >= 14] <- 2
  
  z <- rnorm(n, mean=50, sd=pmax(0.1, x))
  return(data.frame(x=x, y=y, z=z, true_label=labels))
}

# 8. Heavy Tailed Noise (Laplace Approximation)
rlaplace <- function(n, mu=0, b=1) {
  u <- runif(n, -0.5, 0.5)
  mu - b * sign(u) * log(1 - 2 * abs(u))
}

gen_heavy_tails <- function(n=2000) {
  X <- numeric(0); Y <- numeric(0); Z <- numeric(0); L <- numeric(0)
  centers <- matrix(c(-5,0, 5,0), ncol=2, byrow=TRUE)
  pts_per <- n %/% 2
  
  # Cluster 0: Gaussian
  X <- c(X, rnorm(pts_per, centers[1,1], 1.5))
  Y <- c(Y, rnorm(pts_per, centers[1,2], 1.5))
  Z <- c(Z, rnorm(pts_per, 50, 5))
  L <- c(L, rep(0, pts_per))
  
  # Cluster 1: Laplace (Heavy Tail)
  X <- c(X, rnorm(pts_per, centers[2,1], 1.5))
  Y <- c(Y, rnorm(pts_per, centers[2,2], 1.5))
  Z <- c(Z, rlaplace(pts_per, 50, 5))
  L <- c(L, rep(1, pts_per))
  
  return(data.frame(x=X, y=Y, z=Z, true_label=L))
}

# 9. Checkerboard
gen_checkerboard <- function(n=2000) {
  x <- runif(n, 0, 10)
  y <- runif(n, 0, 10)
  z <- numeric(n)
  labels <- numeric(n)
  
  mask_bl <- (x < 5) & (y < 5)
  mask_br <- (x >= 5) & (y < 5)
  mask_tl <- (x < 5) & (y >= 5)
  mask_tr <- (x >= 5) & (y >= 5)
  
  labels[mask_bl] <- 0; z[mask_bl] <- rnorm(sum(mask_bl), 50, 1.0)
  labels[mask_br] <- 1; z[mask_br] <- rnorm(sum(mask_br), 50, 20.0)
  labels[mask_tl] <- 2; z[mask_tl] <- rnorm(sum(mask_tl), 50, 20.0)
  labels[mask_tr] <- 3; z[mask_tr] <- rnorm(sum(mask_tr), 50, 1.0)
  
  return(data.frame(x=x, y=y, z=z, true_label=labels))
}

# 10. Concentric Rings
gen_rings <- function(n=2000) {
  theta <- runif(n, 0, 2*pi)
  pts_per <- n %/% 2
  
  # Ring 1
  r1 <- rnorm(pts_per, 3, 0.5)
  x1 <- r1 * cos(theta[1:pts_per])
  y1 <- r1 * sin(theta[1:pts_per])
  z1 <- rnorm(pts_per, 50, 2)
  l1 <- rep(0, pts_per)
  
  # Ring 2
  r2 <- rnorm(pts_per, 8, 0.5)
  x2 <- r2 * cos(theta[(pts_per+1):n])
  y2 <- r2 * sin(theta[(pts_per+1):n])
  z2 <- rnorm(pts_per, 50, 15)
  l2 <- rep(1, pts_per)
  
  return(data.frame(x=c(x1,x2), y=c(y1,y2), z=c(z1,z2), true_label=c(l1,l2)))
}

# Registry of Datasets
datasets <- list(
  "1_Baseline_Blobs" = gen_baseline_blobs,
  "2_Sharp_Voronoi" = gen_sharp_voronoi,
  "3_Density_Imbalance" = gen_density_imbalance,
  "4_Variance_Imbalance" = gen_variance_imbalance,
  "5_Anisotropic" = gen_anisotropic,
  "6_Radial_Hetero" = gen_radial_hetero,
  "7_Linear_Gradient" = gen_linear_gradient,
  "8_Heavy_Tails" = gen_heavy_tails,
  "9_Checkerboard" = gen_checkerboard,
  "10_Concentric_Rings" = gen_rings
)

# ==========================================
# 2. CLIC WRAPPER
# ==========================================
run_clic_custom <- function(data_matrix, R=1000, B=200, Th=1) {
  n <- nrow(data_matrix)
  
  # Adjust L1/L2 if you expect more clusters, e.g. L1=10
  fit <- gibbs_markov(
    n = n,
    X = data_matrix,
    gamma1 = 1, gamma2 = 1,
    mu01 = 0, sigma01 = 1,
    mu02 = 0, sigma02 = 1,
    alpha1 = 1, beta1 = 1,
    alpha2 = 1, beta2 = 1,
    L1 = 10, L2 = 10, 
    a_rho = 1, b_rho = 1,
    R = R
  )
  
  c2 <- fit$c2[-(1:B),]
  ind <- seq(1, nrow(c2), by = Th)
  c2 <- c2[ind,]
  
  psm2 <- mcclust::comp.psm(c2)
  # Fallback to minbinder if mcclust.ext::minVI is unavailable
  mv2 <- tryCatch({
    mcclust.ext::minVI(psm2, c2)
  }, error = function(e) {
    mcclust::minbinder(psm2, c2)
  })
  
  return(mv2$cl)
}

# ==========================================
# 3. MAIN EXECUTION LOOP
# ==========================================
summary_results <- data.frame(Dataset=character(), CLIC_ARI=numeric(), Mclust_ARI=numeric(), stringsAsFactors=FALSE)

print("Starting Simulation Suite (10 Datasets)...")

for (name in names(datasets)) {
  cat(paste0("\nProcessing: ", name, " ...\n"))
  
  # 1. Generate Data (Reduced N for speed, e.g. 500. Increase to 2000 for full run)
  df <- datasets[[name]](n=500)
  
  # 2. Prep Data
  X_spatial <- scale(cbind(df$x, df$y))
  X_value <- scale(matrix(df$z, ncol=1))
  X_combined <- cbind(X_spatial, X_value)
  
  # 3. Run CLIC
  # R=300 for speed. Increase to 3000-5000 for quality.
  clic_labels <- run_clic_custom(X_combined, R=300, B=100)
  df$clic_labels <- clic_labels
  
  # 4. Run Mclust
  mcl <- Mclust(X_combined, G=1:10, verbose=FALSE)
  df$mclust_labels <- mcl$classification
  
  # 5. Calculate Metrics
  ari_clic <- adjustedRandIndex(df$true_label, df$clic_labels)
  ari_mcl <- adjustedRandIndex(df$true_label, df$mclust_labels)
  
  summary_results[nrow(summary_results) + 1,] <- list(name, ari_clic, ari_mcl)
  
  # 6. Generate & Save Plot
  p_true <- ggplot(df, aes(x=x, y=y, color=as.factor(true_label))) +
    geom_point(alpha=0.6, size=1) + theme_minimal() +
    scale_color_viridis_d() + 
    labs(title="Ground Truth", color="Cluster") + theme(legend.position="none")
  
  p_clic <- ggplot(df, aes(x=x, y=y, color=as.factor(clic_labels))) +
    geom_point(alpha=0.6, size=1) + theme_minimal() +
    scale_color_viridis_d() + 
    labs(title=paste0("CLIC (ARI: ", round(ari_clic, 2), ")"), color="Cluster") + theme(legend.position="none")
  
  p_mcl <- ggplot(df, aes(x=x, y=y, color=as.factor(mclust_labels))) +
    geom_point(alpha=0.6, size=1) + theme_minimal() +
    scale_color_viridis_d() + 
    labs(title=paste0("Mclust (ARI: ", round(ari_mcl, 2), ")"), color="Cluster") + theme(legend.position="none")
  
  plot_file <- paste0("eval_", name, ".png")
  png(plot_file, width=1200, height=400, res=100)
  grid.arrange(p_true, p_clic, p_mcl, nrow=1, top=name)
  dev.off()
  cat(paste0("  -> Saved: ", plot_file, "\n"))
}

# ==========================================
# 4. FINAL SUMMARY
# ==========================================
print("\n==========================================")
print("FINAL RESULTS SUMMARY")
print("==========================================")
print(summary_results)
write.csv(summary_results, "final_summary_results.csv", row.names=FALSE)
print("Summary saved to final_summary_results.csv")