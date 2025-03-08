"""
Configuration settings for the IPL Data Explorer Streamlit application.
This module centralizes all configuration settings, constants, and paths.
"""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / 'data'

# App settings
APP_TITLE = "IPL Data Explorer 🏏"
APP_ICON = "🏏"
APP_LAYOUT = "wide"
INITIAL_SIDEBAR_STATE = "expanded"

# Data source information
DATA_SOURCE_URL = "https://cricsheet.org/"
DATA_SOURCE_NAME = "CricSheet"

# Chart settings
CHART_HEIGHT = 400
CHART_HEIGHT_MOBILE = 350
CHART_MARGIN_DESKTOP = dict(t=60, l=50, r=50, b=50)
CHART_MARGIN_MOBILE = dict(t=100, l=30, r=30, b=50)

# Cache settings
CACHE_TTL = 3600  # Time to live for cached data in seconds

# Navigation
NAVIGATION_SECTIONS = [
    "Overview", 
    "Team Analysis", 
    "Player Analysis", 
    "Match Analysis", 
    "Season Analysis", 
    "Venue Analysis", 
    "Dream Team Analysis"
]

# File paths for precomputed data
PRECOMPUTED_DATA = {
    'matches_per_season': DATA_DIR / "overview_matches_per_season.parquet",
    'avg_runs_by_season': DATA_DIR / "overview_avg_runs_by_season.parquet",
    'avg_wickets_by_season': DATA_DIR / "overview_avg_wickets_by_season.parquet",
    'team_participation': DATA_DIR / "overview_team_participation.parquet",
} 