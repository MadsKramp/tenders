# -*- coding: utf-8 -*-

"""
Field preparation module for spend analysis

This module contains utility functions for processing field names in the table, making them harmonized."""

import pandas as pd
import re
from difflib import SequenceMatcher
from collections import defaultdict

# ------------------------------------------------------------------
# Optional NLTK import with graceful fallback
# ------------------------------------------------------------------
try:
    from nltk.tokenize import word_tokenize  # type: ignore
    from nltk.corpus import stopwords  # type: ignore
    import nltk  # type: ignore
    _NLTK_AVAILABLE = True
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
    except LookupError:
        # Attempt silent downloads only if environment permits
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
        except Exception:
            pass
except ImportError:
    _NLTK_AVAILABLE = False
    # Fallback lightweight tokenizer & stopwords
    def word_tokenize(text):  # minimal regex-based tokenizer
        return re.findall(r"[A-Za-z]+", text)
    class _StopwordsWrapper:
        def words(self, lang):
            return set()
    stopwords = _StopwordsWrapper()

def preprocess_field_name(text):
    """
    Preprocess text for better field names in dataframes.
    
    Example: crm_main_group_vendor -> Group Vendor
    Example: year_authorization -> Year
    Example: purchase_amount_eur -> Purchase Amount Eur
    """
    # Replace underscores with spaces, but preserve numbers
    import re
    text = re.sub(r'_+', ' ', text)

    # Tokenize and remove stopwords (fallback if NLTK unavailable)
    # Preserve tokens with numbers (e.g., 'Class3')
    tokens = re.findall(r'[A-Za-z]+\d*|\d+', text)
    try:
        stop_words = set(stopwords.words('english')) if _NLTK_AVAILABLE else set()
    except Exception:
        stop_words = set()
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]

    # Capitalize each word, preserve numbers
    processed_text = ' '.join(word[0].upper() + word[1:] if word else '' for word in filtered_tokens)

    return processed_text

def preprocess_field_names(columns):
    """Vectorized wrapper to preprocess a list/iterable of column names.

    Args:
        columns (Iterable[str]): Original column names.
    Returns:
        list[str]: Preprocessed names.
    """
    return [preprocess_field_name(c) for c in columns]

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

    # Only consider columns with unique names for merging
    col_counts = pd.Series(columns).value_counts()
    unique_columns = [col for col in columns if col_counts[col] == 1]

    import re
    class_pattern = re.compile(r"^Class\d+$", re.IGNORECASE)
    for i in range(len(unique_columns)):
        for j in range(i + 1, len(unique_columns)):
            # Never merge columns named 'Class' followed by a digit (e.g., Class2, Class3, Class4)
            if class_pattern.match(unique_columns[i]) or class_pattern.match(unique_columns[j]):
                continue
            if similar(unique_columns[i], unique_columns[j]) > threshold:
                to_merge[unique_columns[i]].append(unique_columns[j])

    for main_col, similar_cols in to_merge.items():
        for col in similar_cols:
            if col in df.columns:
                # Only merge if both are Series (not DataFrame)
                if isinstance(df[main_col], pd.Series) and isinstance(df[col], pd.Series):
                    df[main_col] = df[main_col].combine_first(df[col])
                    df.drop(columns=[col], inplace=True)
                else:
                    # Skip if either is a DataFrame (still duplicate columns)
                    print(f"[harmonize_field_names] Skipping merge for '{main_col}' and '{col}' due to duplicate columns.")
    return df
def prepare_field_names(df):
    """Prepare and harmonize field names in the dataframe.

    Falls back to regex tokenization if NLTK is not installed.
    """
    df.columns = [preprocess_field_name(col) for col in df.columns]
    df = harmonize_field_names(df)
    if not _NLTK_AVAILABLE:
        print("⚠️ NLTK not installed; used simple regex tokenization for field name prep.")
    return df
