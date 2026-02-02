library(reticulate)
library(Rcpp)
# Source the CLIC functions as defined in your README
sourceCpp("rcppfuncts/sampling.cpp")
sourceCpp("rcppfuncts/postprocessing.cpp")

np <- import("numpy")
data_dir <- "/usr/xtmp/gr90/Spatial/kdd/clis/data/"
data_files <- list.files(data_dir, pattern = "\\.npz$")

for (f_name in data_files) {
    cat("\n--- Running CLIC on:", f_name, "---\n")
    
    # Load data from NPZ
    loader <- np$load(file.path(data_dir, f_name))
    
    # Format data for CLIC (assuming CLIC needs a matrix or data frame)
    # Adjust indexing based on CLIC's expected input structure
    X_coords <- cbind(loader$f[["x"]], loader$f[["y"]])
    Z_target <- loader$f[["z"]]
    true_labels <- loader$f[["labels"]]
    
    # Run CLIC sampling (Placeholder for actual CLIC function call)
    # result <- sample_clic(X_coords, Z_target) 
    
    # Save results for comparison with CLIS
    # save_results(result, f_name)
}