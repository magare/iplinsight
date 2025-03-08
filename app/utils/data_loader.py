import pandas as pd
import streamlit as st
import os
from pathlib import Path
import logging
from typing import Tuple, Dict, Any, Optional
import sys

# Add parent directory to path to allow importing from config
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import DATA_DIR, CACHE_TTL

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@st.cache_data(ttl=CACHE_TTL)
def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and cache the IPL dataset.
    
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Matches and deliveries dataframes
    
    Raises:
        FileNotFoundError: If data files are not found
        Exception: For other loading errors
    """
    try:
        logger.info("Loading IPL dataset from parquet files")
        matches_df = pd.read_parquet(DATA_DIR / 'matches.parquet')
        deliveries_df = pd.read_parquet(DATA_DIR / 'deliveries.parquet')
        logger.info(f"Successfully loaded data: {len(matches_df)} matches, {len(deliveries_df)} deliveries")
        return matches_df, deliveries_df
    except FileNotFoundError as e:
        logger.error(f"Data file not found: {e}")
        raise FileNotFoundError(f"Required data files not found in {DATA_DIR}. Please ensure the data files exist.") from e
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise Exception(f"Failed to load IPL dataset: {e}") from e

@st.cache_data(ttl=CACHE_TTL)
def load_matches_data() -> pd.DataFrame:
    """
    Load and cache the IPL matches dataset.
    
    Returns:
        pd.DataFrame: Matches dataframe
    """
    try:
        return pd.read_parquet(DATA_DIR / 'matches.parquet')
    except Exception as e:
        logger.error(f"Error loading matches data: {e}")
        st.error(f"Failed to load matches data: {e}")
        return pd.DataFrame()  # Return empty dataframe instead of None

@st.cache_data(ttl=CACHE_TTL)
def load_deliveries_data() -> pd.DataFrame:
    """
    Load and cache the IPL deliveries dataset.
    
    Returns:
        pd.DataFrame: Deliveries dataframe
    """
    try:
        return pd.read_parquet(DATA_DIR / 'deliveries.parquet')
    except Exception as e:
        logger.error(f"Error loading deliveries data: {e}")
        st.error(f"Failed to load deliveries data: {e}")
        return pd.DataFrame()  # Return empty dataframe instead of None

def format_large_number(num: Any) -> str:
    """
    Format large numbers with commas.
    
    Args:
        num: Number to format
        
    Returns:
        str: Formatted number string
    """
    if num is None:
        return "N/A"
    try:
        return f"{int(num):,}"
    except (ValueError, TypeError):
        try:
            return f"{float(num):,.2f}"
        except (ValueError, TypeError):
            return str(num)

@st.cache_data(ttl=CACHE_TTL)
def calculate_basic_stats(matches_df: pd.DataFrame, deliveries_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate basic tournament statistics.
    
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
        stats = {
            'total_matches': len(matches_df),
            'total_seasons': matches_df['season'].nunique(),
            'total_teams': pd.concat([matches_df['team1'], matches_df['team2']]).nunique(),
            'total_venues': matches_df['venue'].nunique(),
            'total_cities': matches_df['city'].nunique(),
            'total_runs': deliveries_df['total_runs'].sum(),
            'total_boundaries': len(deliveries_df[deliveries_df['batsman_runs'].isin([4, 6])]),
            'total_sixes': len(deliveries_df[deliveries_df['batsman_runs'] == 6]),
            'total_wickets': deliveries_df['is_wicket'].sum(),
            'total_overs': len(deliveries_df) / 6
        }
        
        # Calculate averages
        stats['avg_runs_per_match'] = stats['total_runs'] / stats['total_matches']
        stats['avg_wickets_per_match'] = stats['total_wickets'] / stats['total_matches']
        
        return stats
    except Exception as e:
        logger.error(f"Error calculating basic stats: {e}")
        return {}

@st.cache_data(ttl=CACHE_TTL)
def load_precomputed_data(file_path: Path) -> pd.DataFrame:
    """
    Load precomputed data from parquet file.
    
    Args:
        file_path: Path to the parquet file
        
    Returns:
        pd.DataFrame: Loaded dataframe or empty dataframe if file not found
    """
    try:
        if file_path.exists():
            return pd.read_parquet(file_path)
        else:
            logger.warning(f"Precomputed file not found: {file_path}")
            return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error loading precomputed data from {file_path}: {e}")
        return pd.DataFrame() 