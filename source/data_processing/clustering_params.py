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
