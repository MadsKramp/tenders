"""
Field preparation module for spend analysis

This module contains utility functions for processing field names in the table, making them harmonized."""

import pandas as pd
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
from difflib import SequenceMatcher
from collections import defaultdict

# Download NLTK data if not already present
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

def preprocess_field_name(text):
    """
    Preprocess text for better field names in dataframes.
    
    Example: crm_main_group_vendor -> Group Vendor
    Example: year_authorization -> Year
    Example: purchase_amaunt_eur -> Purchase Amount Eur
    """
    # Remove common prefixes
    prefixes = ['crm_main_', 'year_', 'purchase_', 'amount_']
    for prefix in prefixes:
        if text.startswith(prefix):
            text = text[len(prefix):]
            break

    # Replace underscores with spaces
    text = text.replace('_', ' ')

    # Tokenize and remove stopwords
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]

    # Capitalize each word
    processed_text = ' '.join(word.capitalize() for word in filtered_tokens)

    return processed_text

def harmonize_field_names(df, threshold=0.8):
    """
    Harmonize field names in the dataframe by merging similar column names.
    
    Args:
        df (pd.DataFrame): Input dataframe with potentially inconsistent column names.
        threshold (float): Similarity threshold for merging column names.
    Returns:
        pd.DataFrame: Dataframe with harmonized column names.
    """
    def similar(a, b):
        return SequenceMatcher(None, a, b).ratio()

    columns = df.columns.tolist()
    to_merge = defaultdict(list)

    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            if similar(columns[i], columns[j]) > threshold:
                to_merge[columns[i]].append(columns[j])

    for main_col, similar_cols in to_merge.items():
        for col in similar_cols:
            if col in df.columns:
                df[main_col] = df[main_col].combine_first(df[col])
                df.drop(columns=[col], inplace=True)

    return df
def prepare_field_names(df):
    """Prepare and harmonize field names in the dataframe."""
    # Preprocess field names
    df.columns = [preprocess_field_name(col) for col in df.columns]
    
    # Harmonize field names
    df = harmonize_field_names(df)
    
    return df
