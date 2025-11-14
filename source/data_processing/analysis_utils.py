"""
Analysis Utilities for Spend Analysis

This module contains utility functions for the fasteners analysis notebook.
It includes data fetching, preprocessing, visualization, and analytical functions
to support the main analysis workflow.
"""
import pandas as pd
import numpy as np
from collections import Counter
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from source.db_connect import BigQueryConnector
from source.db_connect.sql_queries import get_query

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get project and dataset configuration from environment variables
PROJECT_ID = os.getenv('PROJECT_ID')
DATASET_ID = os.getenv('DATASET_ID')
TABLE_ID = os.getenv('TABLE_ID')
PROGRESS_TABLE_ID = os.getenv('PROGRESS_TABLE_ID')
CLUSTERING_TABLE_ID = os.getenv('CLUSTERING_TABLE_ID')

purchase_data_table = f"{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}"

def fetch_purchase_data(bq_client: BigQueryConnector, limit: int = None) -> pd.DataFrame:
    """
    Fetch purchase data from BigQuery, aggragted on product level
    Returns:
    pd.DataFrame: DataFrame with metrics aggregated on product level
    """
    print("Fetching purchase data from BigQuery...")
    query = get_query('fetch_purchase_data.sql').format(purchase_data_table=purchase_data_table)
    if limit is not None:
        query += f" LIMIT {limit}"
    df = bq_client.query(query)
    if df is None:
        raise RuntimeError("BigQuery query returned no results; check credentials, project, and table access.")
    return df

def preprocess_detailed_data(df):
    """
    Preprocess the detailed data for analysis
    
    Args:
        df (pd.DataFrame): Raw data from BigQuery
        
    Returns:
        pd.DataFrame: Preprocessed data
    """
    print("🔧 Preprocessing detailed data...")
    
    original_len = len(df)
    
    # Check for null or empty values in all columns
    print("📊 Checking for missing values...")
    for column in df.columns:
        null_count = df[column].isnull().sum()
        if df[column].dtype == 'object':
            empty_count = (df[column].astype(str).str.strip() == '').sum()
            total_missing = null_count + empty_count
            if total_missing > 0:
                print(f"  {column}: {total_missing} missing values")
        else:
            if null_count > 0:
                print(f"  {column}: {null_count} missing values")

# Remove rows where PLMStatusGlobal contains numeric figurs '500' or '600' or '700'
    df = df[~df['PLMStatusGlobal'].astype(str).str.contains(r'500|600|700', na=False)]
    cleaned_len = len(df)
    print(f"Removed {original_len - cleaned_len} rows based on PLMStatusGlobal filter.")
    return df

def create_dashboard_distributions(df, columns_to_plot, outlier_columns=['q95_quantity', 'total_orders']):
    """
    Create a dashboard to visualize distributions of Group Vendors in the DataFrame.
    
    Args:
        df (pd.DataFrame): Data to analyze
        columns_to_plot (list): List of column names to plot
        outlier_columns (list): Columns to apply outlier removal
    """
    print("📊 Creating dashboard for distributions...")
    
    # Remove outliers based on 95th percentile
    for col in outlier_columns:
        if col in df.columns:
            threshold = df[col].quantile(0.95)
            df = df[df[col] <= threshold]
            print(f"  Removed outliers in {col} above {threshold}")
    
    num_plots = len(columns_to_plot)
    cols = 3
    rows = (num_plots + cols - 1) // cols
    
    plt.figure(figsize=(5 * cols, 4 * rows))
    
    for i, column in enumerate(columns_to_plot):
        plt.subplot(rows, cols, i + 1)
        sns.histplot(df[column].dropna(), kde=True, bins=30)
        plt.title(f'Distribution of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.show()

def create_class3_countryoforigin_relationship_dashboard(df, columns_to_plot, outlier_columns=['q95_quantity', 'total_orders']):
    """
    Create a dashboard showing the relationship Class3 and Country of Origin using boxplots.
    
    Args:
        df (pd.DataFrame): Data to analyze
        columns_to_plot (list): List of columns to plot against Country of Origin
        outlier_columns (list): Columns to apply outlier removal
    """
    print("📊 Creating Class3 vs Country of Origin relationship dashboard...")
    
    # Remove outliers based on 95th percentile
    for col in outlier_columns:
        if col in df.columns:
            threshold = df[col].quantile(0.95)
            df = df[df[col] <= threshold]
            print(f"  Removed outliers in {col} above {threshold}")
    
    num_plots = len(columns_to_plot)
    cols = 3
    rows = (num_plots + cols - 1) // cols
    
    plt.figure(figsize=(5 * cols, 4 * rows))
    
    for i, column in enumerate(columns_to_plot):
        plt.subplot(rows, cols, i + 1)
        sns.boxplot(x='Class3', y=column, data=df)
        plt.title(f'{column} by Class3')
        plt.xlabel('Class3')
        plt.ylabel(column)
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()

def analyze_spend_by_group_vendor(df):
    """
    Analyze spend by Group Vendor and create visualizations.
    
    Args:
        df (pd.DataFrame): Data to analyze
    """
    print("📊 Analyzing spend by Group Vendor...")
    
    spend_by_vendor = df.groupby('GroupVendor')['TotalSpend'].sum().reset_index()
    top_vendors = spend_by_vendor.sort_values(by='TotalSpend', ascending=False).head(10)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='TotalSpend', y='GroupVendor', data=top_vendors, palette='viridis')
    plt.title('Top 10 Group Vendors by Total Spend')
    plt.xlabel('Total Spend')
    plt.ylabel('Group Vendor')
    plt.tight_layout()
    plt.show()

    # Basic statistics
    print("Basic statistics of Total Spend by Group Vendor:")
    print(spend_by_vendor['TotalSpend'].describe())

def analyze_spend_trends_by_group_vendor_and_countryoforigin(df):
    """
    Analyze spend trends by Group Vendor and Country of Origin.
    
    Args:
        df (pd.DataFrame): Data to analyze
    """
    print("📊 Analyzing spend trends by Group Vendor and Country of Origin...")
    
    spend_trends = df.groupby(['GroupVendor', 'CountryOfOrigin'])['TotalSpend'].sum().reset_index()
    
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x='GroupVendor', y='TotalSpend', hue='CountryOfOrigin', data=spend_trends, palette='tab10', s=100)
    plt.title('Spend Trends by Group Vendor and Country of Origin')
    plt.xlabel('Group Vendor')
    plt.ylabel('Total Spend')
    plt.xticks(rotation=45)
    plt.legend(title='Country of Origin', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

def analyze_spend_by_productnumber(df):
    """
    Analyze spend by Product Number and create visualizations.
    
    Args:
        df (pd.DataFrame): Data to analyze
    """
    print("📊 Analyzing spend by Product Number...")
    
    spend_by_product = df.groupby('ProductNumber')['TotalSpend'].sum().reset_index()
    top_products = spend_by_product.sort_values(by='TotalSpend', ascending=False).head(10)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='TotalSpend', y='ProductNumber', data=top_products, palette='magma')
    plt.title('Top 10 Products by Total Spend')
    plt.xlabel('Total Spend')
    plt.ylabel('Product Number')
    plt.tight_layout()
    plt.show()

    # Basic statistics
    print("Basic statistics of Total Spend by Product Number:")
    print(spend_by_product['TotalSpend'].describe())