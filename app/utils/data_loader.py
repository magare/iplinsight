"""
Data loading utilities for the IPL Data Explorer app.
This module provides functions for loading and caching data.
"""

import pandas as pd
import streamlit as st
import os
from pathlib import Path
import logging
from typing import Tuple, Dict, Any, Optional, List, Union, Callable
import sys
import time

# Add parent directory to path to allow importing from config
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import DATA_DIR, CACHE_TTL

# Configure logging
logger = logging.getLogger(__name__)

# Custom exception class for data loading errors
class DataLoadingError(Exception):
    """Exception raised for errors in data loading operations."""
    pass

@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and cache the IPL dataset with performance optimization.
    
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Matches and deliveries dataframes
    
    Raises:
        DataLoadingError: If data loading fails
    """
    try:
        logger.info("Loading IPL dataset from parquet files")
        start_time = time.time()
        
        # Using parquet for faster loading
        matches_file = DATA_DIR / 'matches.parquet'
        deliveries_file = DATA_DIR / 'deliveries.parquet'
        
        if not matches_file.exists() or not deliveries_file.exists():
            logger.error(f"Required data files not found in {DATA_DIR}")
            raise DataLoadingError(f"Required data files not found in {DATA_DIR}")
        
        matches_df = pd.read_parquet(matches_file)
        deliveries_df = pd.read_parquet(deliveries_file)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Successfully loaded data in {elapsed_time:.2f}s: {len(matches_df)} matches, {len(deliveries_df)} deliveries")
        
        return matches_df, deliveries_df
        
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise DataLoadingError(f"Failed to load IPL dataset: {e}") from e

@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def load_matches_data() -> pd.DataFrame:
    """
    Load and cache the IPL matches dataset.
    
    Returns:
        pd.DataFrame: Matches dataframe
    
    Raises:
        DataLoadingError: If data loading fails
    """
    try:
        start_time = time.time()
        matches_file = DATA_DIR / 'matches.parquet'
        
        if not matches_file.exists():
            logger.error(f"Matches data file not found: {matches_file}")
            raise DataLoadingError(f"Matches data file not found: {matches_file}")
            
        matches_df = pd.read_parquet(matches_file)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Successfully loaded matches data in {elapsed_time:.2f}s: {len(matches_df)} matches")
        
        return matches_df
        
    except Exception as e:
        logger.error(f"Error loading matches data: {e}")
        raise DataLoadingError(f"Failed to load matches data: {e}") from e

@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def load_deliveries_data() -> pd.DataFrame:
    """
    Load and cache the IPL deliveries dataset.
    
    Returns:
        pd.DataFrame: Deliveries dataframe
    
    Raises:
        DataLoadingError: If data loading fails
    """
    try:
        start_time = time.time()
        deliveries_file = DATA_DIR / 'deliveries.parquet'
        
        if not deliveries_file.exists():
            logger.error(f"Deliveries data file not found: {deliveries_file}")
            raise DataLoadingError(f"Deliveries data file not found: {deliveries_file}")
            
        deliveries_df = pd.read_parquet(deliveries_file)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Successfully loaded deliveries data in {elapsed_time:.2f}s: {len(deliveries_df)} deliveries")
        
        return deliveries_df
        
    except Exception as e:
        logger.error(f"Error loading deliveries data: {e}")
        raise DataLoadingError(f"Failed to load deliveries data: {e}") from e

def format_large_number(num: Any) -> str:
    """
    Format large numbers with commas or abbreviate very large numbers.
    
    Args:
        num: Number to format
        
    Returns:
        str: Formatted number string
    """
    if num is None:
        return "N/A"
    
    try:
        n = float(num)
        if n >= 1_000_000:
            return f"{n/1_000_000:.1f}M"
        elif n >= 1_000:
            return f"{n/1_000:.1f}K"
        elif n.is_integer():
            return f"{int(n):,}"
        else:
            return f"{n:,.2f}"
    except (ValueError, TypeError):
        return str(num)

@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def calculate_basic_stats(matches_df: pd.DataFrame, deliveries_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate basic tournament statistics with optimized performance.
    
    Args:
        matches_df: Matches dataframe
        deliveries_df: Deliveries dataframe
        
    Returns:
        Dict[str, Any]: Dictionary of calculated statistics
    """
    if matches_df.empty or deliveries_df.empty:
        logger.warning("Empty dataframes provided to calculate_basic_stats")
        return {}
        
    try:
        start_time = time.time()
        
        # Calculate team stats efficiently
        team_cols = ['team1', 'team2']
        all_teams = pd.concat([
            matches_df[col] for col in team_cols if col in matches_df.columns
        ]).unique()
        
        # Use vectorized operations wherever possible for better performance
        stats = {
            'total_matches': len(matches_df),
            'total_seasons': matches_df['season'].nunique(),
            'total_teams': len(all_teams),
            'total_venues': matches_df['venue'].nunique(),
            'total_cities': matches_df['city'].nunique() if 'city' in matches_df.columns else 0,
            'total_runs': deliveries_df['total_runs'].sum() if 'total_runs' in deliveries_df.columns else 0,
            'total_boundaries': len(deliveries_df[deliveries_df['batsman_runs'].isin([4, 6])]) if 'batsman_runs' in deliveries_df.columns else 0,
            'total_sixes': len(deliveries_df[deliveries_df['batsman_runs'] == 6]) if 'batsman_runs' in deliveries_df.columns else 0,
            'total_wickets': deliveries_df['is_wicket'].sum() if 'is_wicket' in deliveries_df.columns else 0,
        }
        
        # Calculate derived stats
        if 'total_matches' in stats and stats['total_matches'] > 0:
            if 'total_runs' in stats:
                stats['avg_runs_per_match'] = stats['total_runs'] / stats['total_matches']
            if 'total_wickets' in stats:
                stats['avg_wickets_per_match'] = stats['total_wickets'] / stats['total_matches']
        
        # Calculate overs safely (avoid division by zero)
        if len(deliveries_df) > 0:
            stats['total_overs'] = len(deliveries_df) / 6
        
        elapsed_time = time.time() - start_time
        logger.debug(f"Calculated basic stats in {elapsed_time:.2f}s")
        
        return stats
        
    except Exception as e:
        logger.error(f"Error calculating basic stats: {e}")
        return {}

@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def load_precomputed_data(file_path: Path) -> pd.DataFrame:
    """
    Load precomputed data from parquet file with optimized error handling.
    
    Args:
        file_path: Path to the parquet file
        
    Returns:
        pd.DataFrame: Loaded dataframe or empty dataframe if file not found
    """
    try:
        if not file_path.exists():
            logger.warning(f"Precomputed file not found: {file_path}")
            return pd.DataFrame()
            
        start_time = time.time()
        data = pd.read_parquet(file_path)
        elapsed_time = time.time() - start_time
        
        logger.debug(f"Loaded precomputed data from {file_path} in {elapsed_time:.2f}s: {len(data)} rows")
        return data
        
    except Exception as e:
        logger.error(f"Error loading precomputed data from {file_path}: {e}")
        return pd.DataFrame()

def data_loader_with_retry(loader_func: Callable, max_retries: int = 3, *args, **kwargs) -> Any:
    """
    Wrapper function to attempt data loading with retries.
    
    Args:
        loader_func: Data loading function to call
        max_retries: Maximum number of retry attempts
        *args, **kwargs: Arguments to pass to the loader function
        
    Returns:
        Any: The result of the loader function
        
    Raises:
        DataLoadingError: If all retry attempts fail
    """
    retry_count = 0
    last_error = None
    
    while retry_count < max_retries:
        try:
            return loader_func(*args, **kwargs)
        except Exception as e:
            retry_count += 1
            last_error = e
            wait_time = 2 ** retry_count  # Exponential backoff
            logger.warning(f"Retry {retry_count}/{max_retries} after error: {e}. Waiting {wait_time}s")
            time.sleep(wait_time)
    
    logger.error(f"Failed to load data after {max_retries} retries.")
    raise DataLoadingError(f"Failed to load data after {max_retries} retries: {last_error}") 