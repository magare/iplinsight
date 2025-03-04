import pandas as pd
import streamlit as st
import os
from pathlib import Path

# Get the current directory
current_dir = Path(__file__).resolve().parent
# Navigate to the app/data directory correctly
data_dir = current_dir.parent / 'data'

@st.cache_data
def load_data():
    """Load and cache the IPL dataset."""
    matches_df = pd.read_parquet(data_dir / 'matches.parquet')
    deliveries_df = pd.read_parquet(data_dir / 'deliveries.parquet')
    return matches_df, deliveries_df

@st.cache_data
def load_matches_data():
    """Load and cache the IPL matches dataset."""
    return pd.read_parquet(data_dir / 'matches.parquet')

@st.cache_data
def load_deliveries_data():
    """Load and cache the IPL deliveries dataset."""
    return pd.read_parquet(data_dir / 'deliveries.parquet')

def format_large_number(num):
    """Format large numbers with commas."""
    return f"{num:,}"

def calculate_basic_stats(matches_df, deliveries_df):
    """Calculate basic tournament statistics."""
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