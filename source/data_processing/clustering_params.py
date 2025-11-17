"""
Centralized Parameters for Clustering Analysis

This module provides a single place to define all clustering analysis parameters.
Simple, clean, and easy to modify without unnecessary complexity.
"""

# =============================================================================
# DATA FILTERING PARAMETERS
# =============================================================================

# Minimum number of transactions required per product for analysis
MIN_TRANSACTIONS = 50

# Product filtering
CLASS3_DESCRIPTION = 'Bolts & Nuts'
PRODUCT_DESCRIPTION = 'nut'  # Leave empty '' for all products

# =============================================================================
# FEATURE CONFIGURATION
# =============================================================================

# Optional performance & feature engineering controls
ENABLE_SAMPLING = True            # Use a subset of data for cluster number optimization
SAMPLING_FRACTION = 0.3           # Fraction of rows to sample (ignored if resulting size exceeds MAX_SAMPLE_SIZE)
MAX_SAMPLE_SIZE = 10000           # Hard cap on sample size

USE_MINIBATCH_KMEANS = True       # Use MiniBatchKMeans instead of full KMeans for speed
MINIBATCH_BATCH_SIZE = 512        # Batch size for MiniBatchKMeans
MINIBATCH_MAX_ITER = 100          # Max iterations for MiniBatchKMeans

ENABLE_COLUMN_PRUNING = True      # Remove highly correlated numeric columns prior to scaling
COLUMN_PRUNE_CORRELATION_THRESHOLD = 0.95  # Absolute correlation above which one column of the pair is dropped

USE_PCA = True                    # Apply PCA dimensionality reduction after scaling
PCA_VARIANCE_THRESHOLD = 0.95     # Retain components until cumulative explained variance >= this threshold

# =============================================================================
# SINGLE-COLUMN CLUSTERING (override multi-feature logic)
# =============================================================================
CLUSTER_BY_SINGLE_COLUMN = True          # When True, cluster only using a single numeric column
SINGLE_COLUMN_NAME = 'purchase_amount_eur'  # Column to use when single-column clustering is enabled

# =============================================================================
# PURCHASE PATTERN FEATURES (requires raw transactional data)
# =============================================================================
INCLUDE_PATTERN_FEATURES = True  # Build pct_qty_* distribution columns from raw transactions
INCLUDE_VALUE_PATTERN_FEATURES = True  # Build pct_purchase_amount_eur_* distribution columns from raw transactions


# =============================================================================
# CLUSTERING PARAMETERS
# =============================================================================

# Range of cluster numbers to test during optimization
MAX_CLUSTERS = 10
MIN_CLUSTERS = 2

# Random state for reproducible results
RANDOM_STATE = 42

# Which clustering methods to run
INCLUDE_KMEANS = True
INCLUDE_HIERARCHICAL = True
INCLUDE_DBSCAN = True

# =============================================================================
# DBSCAN SPECIFIC PARAMETERS (optional - auto-optimized if None)
# =============================================================================

DBSCAN_EPS = None  # Auto-optimized if None
DBSCAN_MIN_SAMPLES = None  # Auto-optimized if None

# =============================================================================
# OUTPUT AND VISUALIZATION
# =============================================================================

# Whether to show plots and visualizations
SHOW_VISUALIZATIONS = True

# Whether to export results to Excel files
EXPORT_RESULTS = True

# =============================================================================
# ANALYSIS EXECUTION
# =============================================================================

# Override automatic cluster number selection (None to use optimization)
N_CLUSTERS_OVERRIDE = None
