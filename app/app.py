"""
IPL Data Explorer - Main Application
This is the main entry point for the Streamlit application.
"""

import streamlit as st

# Set page configuration - MUST BE FIRST STREAMLIT COMMAND
st.set_page_config(
    page_title="IPL Data Explorer 🏏",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

import logging
from pathlib import Path
import sys
from typing import Tuple, Dict, Any, Optional
import time

# Import configuration
from config import (
    APP_TITLE, 
    APP_ICON, 
    APP_LAYOUT, 
    INITIAL_SIDEBAR_STATE, 
    NAVIGATION_SECTIONS,
    LOG_LEVEL
)

# Import utilities
from utils.logger import configure_app_logging, get_module_logger
from utils.state_manager import initialize_session_state, get_state, set_state
from utils.error_handler import ErrorBoundary, display_error
from utils.data_loader import (
    load_data, 
    calculate_basic_stats, 
    format_large_number,
    data_loader_with_retry,
    DataLoadingError
)
from utils.chart_utils import init_device_detection
from utils.ui_components import (
    load_css, 
    HeaderComponent,
    SidebarComponent,
    MessageComponent,
    DataDisplayComponent,
    LayoutComponent
)

# Import components
from components.overview import render_overview_section
from components.team_analysis import display_team_analysis
from components.player_analysis import (
    display_batting_analysis,
    display_bowling_analysis,
    display_allrounder_analysis,
    display_head_to_head_analysis
)
from components.match_analysis import (
    display_match_result_analysis,
    display_toss_analysis,
    display_scoring_analysis,
    display_high_low_scoring_analysis,
    display_match_details
)
from components.season_analysis import display_season_analysis
from components.venue_analysis import display_venue_analysis
from components.dream_team_analysis import DreamTeamAnalysis

# Configure logging
configure_app_logging(LOG_LEVEL)
logger = get_module_logger(__name__)

# Initialize application state
def initialize_app():
    """Initialize application state and detect device."""
    # Initialize session state with default values
    initialize_session_state({
        'device_type': 'desktop',
        'data_loaded': False,
        'selected_tab': NAVIGATION_SECTIONS[0],
        'loading_error': None
    })
    
    # Initialize device detection
    device_type = init_device_detection()
    
    # Load CSS
    load_css()
    
    logger.info("Application initialized")

@st.cache_data(show_spinner=False)
def load_cached_data():
    """
    Load and cache the IPL dataset.
    
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Matches and deliveries dataframes
    """
    try:
        start_time = time.time()
        logger.info("Loading cached data")
        
        # Use retry mechanism for more reliable data loading
        matches, deliveries = data_loader_with_retry(load_data)
        
        set_state('data_loaded', True)
        set_state('loading_error', None)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Data loading completed in {elapsed_time:.2f}s")
        
        return matches, deliveries
        
    except Exception as e:
        set_state('loading_error', str(e))
        logger.error(f"Error loading cached data: {e}")
        raise

def render_navigation():
    """Render the navigation sidebar."""
    with ErrorBoundary("Navigation Sidebar"):
        SidebarComponent.sidebar_header()
        
        # Render navigation options
        selected_tab = st.sidebar.radio("", NAVIGATION_SECTIONS)
        set_state('selected_tab', selected_tab)
        
        # Add information at the bottom of the sidebar
        try:
            matches, _ = load_cached_data()
            min_season = int(matches['season'].min())
            max_season = int(matches['season'].max())
            SidebarComponent.sidebar_footer(min_season, max_season)
        except Exception as e:
            logger.error(f"Error displaying sidebar footer: {e}")

# Define component rendering functions to match our new refactored structure
def render_team_analysis_section(matches, deliveries):
    """Render team analysis section using the existing component."""
    with ErrorBoundary("Team Analysis Section"):
        display_team_analysis(matches, deliveries)

def render_player_analysis_section(matches, deliveries):
    """Render player analysis section using the existing components."""
    with ErrorBoundary("Player Analysis Section"):
        st.header("Player Analysis")
        
        # Get device type to determine layout
        device_type = get_state('device_type', 'mobile')
        
        # Define analysis options
        analysis_options = ["Batting Analysis", "Bowling Analysis", "All-Rounder Analysis", "Head-to-Head Analysis"]
        
        if device_type == 'mobile':
            # Use a selectbox for mobile view
            selected_analysis = st.selectbox("Select Analysis", analysis_options)
            
            # Display the selected analysis
            if selected_analysis == "Batting Analysis":
                display_batting_analysis(deliveries)
            elif selected_analysis == "Bowling Analysis":
                display_bowling_analysis(deliveries)
            elif selected_analysis == "All-Rounder Analysis":
                display_allrounder_analysis(deliveries)
            elif selected_analysis == "Head-to-Head Analysis":
                display_head_to_head_analysis(deliveries)
        else:
            # Use tabs for desktop view
            tabs = st.tabs(analysis_options)
            with tabs[0]:
                display_batting_analysis(deliveries)
            with tabs[1]:
                display_bowling_analysis(deliveries)
            with tabs[2]:
                display_allrounder_analysis(deliveries)
            with tabs[3]:
                display_head_to_head_analysis(deliveries)

def render_match_analysis_section(matches, deliveries):
    """Render match analysis section using the existing components."""
    with ErrorBoundary("Match Analysis Section"):
        st.header("Match Analysis")
        st.markdown("""
        This section provides a detailed analysis of match outcomes, toss decisions, and their impact on the game.
        We explore various aspects like victory margins, toss advantage, and scoring patterns.
        """)
        
        # Define analysis options
        match_options = ["Match Details", "Match Results", "Toss Analysis", "Scoring Patterns", "High/Low Scoring Matches"]
        
        # Use a selectbox for both mobile and desktop for consistency
        match_tab = st.selectbox("Select Match Analysis", match_options)
        
        if match_tab == "Match Results":
            display_match_result_analysis(matches)
        elif match_tab == "Toss Analysis":
            display_toss_analysis(matches)
        elif match_tab == "Scoring Patterns":
            display_scoring_analysis(matches, deliveries)
        elif match_tab == "High/Low Scoring Matches":
            display_high_low_scoring_analysis(matches, deliveries)
        elif match_tab == "Match Details":
            display_match_details(matches)

def render_season_analysis_section(matches, deliveries):
    """Render season analysis section using the existing component."""
    with ErrorBoundary("Season Analysis Section"):
        display_season_analysis(matches, deliveries)

def render_venue_analysis_section(matches, deliveries):
    """Render venue analysis section using the existing component."""
    with ErrorBoundary("Venue Analysis Section"):
        display_venue_analysis(matches, deliveries)

def render_dream_team_section(matches, deliveries):
    """Render dream team section using the existing component."""
    with ErrorBoundary("Dream Team Analysis Section"):
        dt_instance = DreamTeamAnalysis()
        dt_instance.create_layout()

def render_section(section: str, matches, deliveries):
    """
    Render the selected section.
    
    Args:
        section: The section to render
        matches: Matches dataframe
        deliveries: Deliveries dataframe
    """
    section_mapping = {
        "Overview": render_overview_section,
        "Team Analysis": render_team_analysis_section,
        "Player Analysis": render_player_analysis_section,
        "Match Analysis": render_match_analysis_section,
        "Season Analysis": render_season_analysis_section,
        "Venue Analysis": render_venue_analysis_section,
        "Dream Team Analysis": render_dream_team_section
    }
    
    if section in section_mapping:
        with ErrorBoundary(f"{section} Section"):
            section_mapping[section](matches, deliveries)
    else:
        logger.error(f"Unknown section: {section}")
        MessageComponent.error(f"Unknown section: {section}")

def main():
    """
    Main application function.
    """
    try:
        # Initialize application
        initialize_app()
        
        # Render navigation
        render_navigation()
        
        # Check if there was a loading error from a previous attempt
        loading_error = get_state('loading_error')
        if loading_error:
            MessageComponent.error(f"Failed to load data: {loading_error}")
            
            # Add retry button
            if st.button("Retry Loading Data"):
                st.experimental_rerun()
            
            st.stop()
        
        # Load data with error handling
        try:
            with st.spinner("Loading data..."):
                matches, deliveries = load_cached_data()
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            MessageComponent.error("Failed to load data. Please try refreshing the page.")
            
            # Add technical details in an expander
            with st.expander("Technical Details", expanded=False):
                st.code(str(e))
            
            st.stop()
        
        # Render the selected section
        selected_tab = get_state('selected_tab')
        render_section(selected_tab, matches, deliveries)
        
    except Exception as e:
        logger.error(f"Unhandled application error: {e}", exc_info=True)
        MessageComponent.error("An unexpected error occurred. Please try refreshing the page.")
        
        # Add technical details in an expander for debugging
        with st.expander("Technical Details", expanded=False):
            st.code(str(e))

if __name__ == "__main__":
    main()
