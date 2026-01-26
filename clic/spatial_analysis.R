# Load required packages
library(sf)
library(Rcpp)
library(RcppArmadillo)
library(mclust)
library(mcclust)
library(mcclust.ext)
library(ggplot2)

# Source CLIC functions
sourceCpp("rcppfuncts/sampling.cpp")
sourceCpp("rcppfuncts/postprocessing.cpp")
source("rfuncts/multiview_dependent.R")

#' Process Spatial Data for CLIC Analysis
#' @param shp_path Path to shapefile
#' @param value_col Name of column containing land values
#' @return List containing processed data and CLIC results
process_spatial_clic <- function(shp_path, value_col) {
  # Read shapefile
  parcels <- st_read(shp_path)
  
  # Calculate centroids and extract coordinates
  centroids <- st_centroid(parcels)
  coords <- st_coordinates(centroids)
  
  # Normalize coordinates and values for better clustering
  coords_scaled <- scale(coords)
  values_scaled <- scale(parcels[[value_col]])
  
  # Prepare data for CLIC
  # View 1: Spatial coordinates
  X1 <- coords_scaled
  # View 2: Land values as 1D data
  X2 <- matrix(values_scaled, ncol=1)
  
  # Combine data for CLIC
  X_combined <- cbind(X1, X2)
  
  # Run CLIC
  results <- multiview_dependent(
    n = nrow(X_combined),
    omega0 = 1/2,
    R = 5000,        # Increase for better results
    B = 1000,        # Increase for better results
    Th = 2,
    eta = sqrt(0.2)  # Adjust based on your data
  )
  
  # Add clustering results back to spatial data
  parcels$cluster <- results$c1mat[,ncol(results$c1mat)]
  
  # Create visualization
  cluster_map <- ggplot(parcels) +
    geom_sf(aes(fill=as.factor(cluster))) +
    scale_fill_viridis_d(name="Cluster") +
    theme_minimal() +
    labs(title="Land Parcel Clusters")
  
  return(list(
    parcels = parcels,
    results = results,
    plot = cluster_map
  ))
}

# Example usage:
results <- process_spatial_clic("/home/users/gr90/projects/Spatial/parcels/parview2025.shp", "LAND_VALUE")
# print(results$plot)