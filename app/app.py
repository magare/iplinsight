"""
IPL Data Explorer - Main Application
This is the main entry point for the Streamlit application.
"""

import streamlit as st
import logging
from pathlib import Path
import sys
from typing import Tuple, Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import configuration
from config import APP_TITLE, APP_ICON, APP_LAYOUT, INITIAL_SIDEBAR_STATE, NAVIGATION_SECTIONS

# Import utilities
from utils.data_loader import load_data, calculate_basic_stats, format_large_number
from utils.chart_utils import init_device_detection
from utils.ui_components import (
    load_css, 
    display_app_header, 
    display_sidebar_header, 
    display_sidebar_footer,
    display_error_message
)

# Import components
from components.overview import (
    plot_tournament_growth,
    display_team_participation,
    display_dataset_info,
    display_key_questions,
    display_data_limitations
)
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

# Set page configuration
st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout=APP_LAYOUT,
    initial_sidebar_state=INITIAL_SIDEBAR_STATE
)

def main():
    """
    Main application function.
    """
    try:
        # Load CSS
        load_css()
        
        # Initialize device detection
        device_type = init_device_detection()
        
        # Load data with error handling
        try:
            with st.spinner("Loading data..."):
                matches, deliveries = load_cached_data()
        except Exception as e:
            display_error_message("Failed to load data. Please try refreshing the page.", e)
            st.stop()
        
        # Set up sidebar navigation
        display_sidebar_header()
        selected_tab = st.sidebar.radio("", NAVIGATION_SECTIONS)
        
        # Add information at the bottom of the sidebar
        try:
            min_season = int(matches['season'].min())
            max_season = int(matches['season'].max())
            display_sidebar_footer(min_season, max_season)
        except Exception as e:
            logger.error(f"Error displaying sidebar footer: {e}")
        
        # Render the selected section
        render_section(selected_tab, matches, deliveries)
        
    except Exception as e:
        logger.error(f"Unhandled application error: {e}", exc_info=True)
        display_error_message("An unexpected error occurred. Please try refreshing the page.")

@st.cache_data(show_spinner=False)
def load_cached_data():
    """
    Load and cache the IPL dataset.
    
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Matches and deliveries dataframes
    """
    logger.info("Loading cached data")
    return load_data()

def render_section(section: str, matches, deliveries):
    """
    Render the selected section.
    
    Args:
        section: The section to render
        matches: Matches dataframe
        deliveries: Deliveries dataframe
    """
    if section == "Overview":
        render_overview(matches, deliveries)
    elif section == "Team Analysis":
        render_team_analysis(matches, deliveries)
    elif section == "Player Analysis":
        render_player_analysis(matches, deliveries)
    elif section == "Match Analysis":
        render_match_analysis(matches, deliveries)
    elif section == "Season Analysis":
        render_season_analysis(matches, deliveries)
    elif section == "Venue Analysis":
        render_venue_analysis(matches, deliveries)
    elif section == "Dream Team Analysis":
        render_dream_team_analysis()

def render_overview(matches, deliveries):
    """
    Render the overview section.
    
    Args:
        matches: Matches dataframe
        deliveries: Deliveries dataframe
    """
    try:
        # Display app header
        display_app_header()
        
        st.header("IPL Data Analysis Overview")
        st.markdown("""
        The Indian Premier League (IPL) has captivated cricket fans worldwide since its inception in 2008. 
        This analysis dives deep into the rich history of IPL, uncovering patterns and insights from over a decade of matches.
        """)
        
        # Calculate basic stats
        stats = calculate_basic_stats(matches, deliveries)
        
        # Display metrics in three columns
        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("Tournament Scale")
            st.metric("Total Matches", format_large_number(stats['total_matches']))
            st.metric("Unique Teams", format_large_number(stats['total_teams']))
            st.metric("Seasons", format_large_number(stats['total_seasons']))
            st.metric("Venues", format_large_number(stats['total_venues']))
            st.metric("Cities", format_large_number(stats['total_cities']))
        with col2:
            st.subheader("Batting Insights")
            st.metric("Total Runs", format_large_number(stats['total_runs']))
            st.metric("Total Boundaries", format_large_number(stats['total_boundaries']))
            st.metric("Total Sixes", format_large_number(stats['total_sixes']))
            try:
                avg_runs = float(stats['avg_runs_per_match'])
                st.metric("Avg. Runs/Match", f"{avg_runs:.1f}")
            except (TypeError, ValueError):
                st.metric("Avg. Runs/Match", "N/A")
        with col3:
            st.subheader("Bowling Insights")
            st.metric("Total Wickets", format_large_number(stats['total_wickets']))
            try:
                total_overs = int(stats['total_overs'])
                st.metric("Total Overs", format_large_number(total_overs))
            except (TypeError, ValueError):
                st.metric("Total Overs", "N/A")
            try:
                avg_wickets = float(stats['avg_wickets_per_match'])
                st.metric("Avg. Wickets/Match", f"{avg_wickets:.1f}")
            except (TypeError, ValueError):
                st.metric("Avg. Wickets/Match", "N/A")
        
        st.subheader("Tournament Growth Over Years")
        plot_tournament_growth(matches, deliveries)
        
        # Team Participation Table
        display_team_participation(matches)
        
        # Key questions and data limitations
        display_key_questions()
        display_data_limitations()
        
        # Dataset structure information (moved to the bottom)
        display_dataset_info()
    except Exception as e:
        logger.error(f"Error rendering overview: {e}")
        display_error_message("Error rendering overview section. Please try refreshing the page.")

def render_team_analysis(matches, deliveries):
    """
    Render the team analysis section.
    
    Args:
        matches: Matches dataframe
        deliveries: Deliveries dataframe
    """
    try:
        display_team_analysis(matches, deliveries)
    except Exception as e:
        logger.error(f"Error rendering team analysis: {e}")
        display_error_message("Error rendering team analysis section. Please try refreshing the page.")

def render_player_analysis(matches, deliveries):
    """
    Render the player analysis section.
    
    Args:
        matches: Matches dataframe
        deliveries: Deliveries dataframe
    """
    try:
        st.header("Player Analysis")
        
        # Get device type to determine layout
        device_type = st.session_state.get('device_type', 'mobile')
        
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
    except Exception as e:
        logger.error(f"Error rendering player analysis: {e}")
        display_error_message("Error rendering player analysis section. Please try refreshing the page.")

def render_match_analysis(matches, deliveries):
    """
    Render the match analysis section.
    
    Args:
        matches: Matches dataframe
        deliveries: Deliveries dataframe
    """
    try:
        st.header("Match Analysis")
        st.markdown("""
        This section provides a detailed analysis of match outcomes, toss decisions, and their impact on the game.
        We explore various aspects like victory margins, toss advantage, and scoring patterns.
        """)
        
        # Get device type to determine layout
        device_type = st.session_state.get('device_type', 'mobile')
        
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
    except Exception as e:
        logger.error(f"Error rendering match analysis: {e}")
        display_error_message("Error rendering match analysis section. Please try refreshing the page.")

def render_season_analysis(matches, deliveries):
    """
    Render the season analysis section.
    
    Args:
        matches: Matches dataframe
        deliveries: Deliveries dataframe
    """
    try:
        display_season_analysis(matches, deliveries)
    except Exception as e:
        logger.error(f"Error rendering season analysis: {e}")
        display_error_message("Error rendering season analysis section. Please try refreshing the page.")

def render_venue_analysis(matches, deliveries):
    """
    Render the venue analysis section.
    
    Args:
        matches: Matches dataframe
        deliveries: Deliveries dataframe
    """
    try:
        display_venue_analysis(matches, deliveries)
    except Exception as e:
        logger.error(f"Error rendering venue analysis: {e}")
        display_error_message("Error rendering venue analysis section. Please try refreshing the page.")

def render_dream_team_analysis():
    """
    Render the dream team analysis section.
    """
    try:
        dt_instance = DreamTeamAnalysis()
        dt_instance.create_layout()
    except Exception as e:
        logger.error(f"Error rendering dream team analysis: {e}")
        display_error_message("Error rendering dream team analysis section. Please try refreshing the page.")

if __name__ == "__main__":
    main()
