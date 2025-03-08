"""
Overview component for the IPL Data Explorer app.
This module provides functions for displaying overview information about the IPL dataset.
"""

import streamlit as st
import plotly.express as px
import pandas as pd
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional

# Add parent directory to path to allow importing from config
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import PRECOMPUTED_DATA
from utils.chart_utils import responsive_plotly_chart, get_neon_color_discrete_sequence
from utils.data_loader import load_precomputed_data
from utils.color_palette import NEON_COLORS, CHART_STYLING

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_overview_data() -> Dict[str, pd.DataFrame]:
    """
    Load pre-computed data for the overview section.
    
    Returns:
        Dict[str, pd.DataFrame]: Dictionary of pre-computed dataframes
    """
    try:
        # Load pre-computed datasets
        matches_per_season = load_precomputed_data(PRECOMPUTED_DATA['matches_per_season'])
        avg_runs_by_season = load_precomputed_data(PRECOMPUTED_DATA['avg_runs_by_season'])
        avg_wickets_by_season = load_precomputed_data(PRECOMPUTED_DATA['avg_wickets_by_season'])
        team_participation = load_precomputed_data(PRECOMPUTED_DATA['team_participation'])
        
        # Rename columns to make them more human-readable
        if not team_participation.empty:
            team_participation = team_participation.rename(columns={
                'Seasons_Played': 'Seasons Played',
                'Total_Matches': 'Total Matches',
                'Wins': 'Wins',
                'Win_Rate': 'Win Rate (%)'
            })
        
        return {
            'matches_per_season': matches_per_season,
            'avg_runs_by_season': avg_runs_by_season,
            'avg_wickets_by_season': avg_wickets_by_season,
            'team_participation': team_participation
        }
    except Exception as e:
        logger.error(f"Error loading overview data: {e}")
        st.error(f"Failed to load overview data. Please try refreshing the page.")
        return {
            'matches_per_season': pd.DataFrame(),
            'avg_runs_by_season': pd.DataFrame(),
            'avg_wickets_by_season': pd.DataFrame(),
            'team_participation': pd.DataFrame()
        }

def plot_tournament_growth(matches_df=None, deliveries_df=None) -> None:
    """
    Plot tournament growth statistics using pre-computed data.
    
    Args:
        matches_df: Optional matches dataframe (not used, kept for backward compatibility)
        deliveries_df: Optional deliveries dataframe (not used, kept for backward compatibility)
    """
    try:
        # Load pre-computed data
        overview_data = load_overview_data()
        matches_per_season = overview_data['matches_per_season']
        avg_runs_by_season = overview_data['avg_runs_by_season']
        avg_wickets_by_season = overview_data['avg_wickets_by_season']
        
        # Check if data is available
        if matches_per_season.empty or avg_runs_by_season.empty or avg_wickets_by_season.empty:
            st.warning("Tournament growth data is not available. Please check the data files.")
            return
        
        col1, col2, col3 = st.columns(3)
        
        # Common chart settings using our neon colors
        chart_colors = [NEON_COLORS[0]]  # Neon green
        secondary_chart_colors = [NEON_COLORS[2]]  # Neon cyan
        tertiary_chart_colors = [NEON_COLORS[1]]  # Neon pink
        
        with col1:
            # Matches per season (using pre-computed data)
            fig = px.bar(matches_per_season, 
                        x='season', 
                        y='matches',
                        title='Number of Matches per Season',
                        text='matches',
                        template=CHART_STYLING['template'],
                        color_discrete_sequence=chart_colors)
            
            fig.update_traces(
                textposition='outside',
                textfont=dict(size=12),
                hovertemplate="Season: %{x}<br>Matches: %{y}<extra></extra>"
            )
            fig.update_layout(
                height=400,
                xaxis_title="Season",
                yaxis=dict(
                    title_text="Number of Matches",
                    gridcolor='rgba(128,128,128,0.1)',
                    tickmode='linear',
                    dtick=10,  # Increase tick spacing from 5 to 10 matches
                    automargin=True
                ),
                xaxis=dict(
                    ticktext=matches_per_season['season'],
                    tickvals=matches_per_season['season'],
                    tickangle=-45,
                    tickfont=dict(size=10)
                ),
                title=dict(
                    font=dict(size=16),
                    y=0.95
                ),
                showlegend=False
            )
            responsive_plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Average runs per match by season (using pre-computed data)
            fig = px.bar(avg_runs_by_season,
                        x='season',
                        y='total_runs',
                        title='Average Runs per Match by Season',
                        text=avg_runs_by_season['total_runs'].round(1),
                        template=CHART_STYLING['template'],
                        color_discrete_sequence=secondary_chart_colors)
            
            fig.update_traces(
                textposition='outside',
                textfont=dict(size=12),
                hovertemplate="Season: %{x}<br>Avg Runs: %{y:.1f}<extra></extra>"
            )
            fig.update_layout(
                height=400,
                xaxis_title="Season",
                yaxis=dict(
                    title_text="Average Runs",
                    gridcolor='rgba(128,128,128,0.1)',
                    tickmode='linear',
                    dtick=20,  # Increase tick spacing from 5 to 20 runs
                    automargin=True
                ),
                xaxis=dict(
                    ticktext=avg_runs_by_season['season'],
                    tickvals=avg_runs_by_season['season'],
                    tickangle=-45,
                    tickfont=dict(size=10)
                ),
                title=dict(
                    font=dict(size=16),
                    y=0.95
                ),
                showlegend=False
            )
            responsive_plotly_chart(fig, use_container_width=True)
        
        with col3:
            # Average wickets per match by season (using pre-computed data)
            fig = px.bar(avg_wickets_by_season,
                        x='season',
                        y='wickets',
                        title='Average Wickets per Match by Season',
                        text=avg_wickets_by_season['wickets'].round(1),
                        template=CHART_STYLING['template'],
                        color_discrete_sequence=tertiary_chart_colors)
            
            fig.update_traces(
                textposition='outside',
                textfont=dict(size=12),
                hovertemplate="Season: %{x}<br>Avg Wickets: %{y:.1f}<extra></extra>"
            )
            fig.update_layout(
                height=400,
                xaxis_title="Season",
                yaxis=dict(
                    title_text="Average Wickets",
                    gridcolor='rgba(128,128,128,0.1)',
                    tickmode='linear',
                    dtick=1,  # Increase tick spacing from 0.5 to 1 wicket
                    automargin=True
                ),
                xaxis=dict(
                    ticktext=avg_wickets_by_season['season'],
                    tickvals=avg_wickets_by_season['season'],
                    tickangle=-45,
                    tickfont=dict(size=10)
                ),
                title=dict(
                    font=dict(size=16),
                    y=0.95
                ),
                showlegend=False
            )
            responsive_plotly_chart(fig, use_container_width=True)
    except Exception as e:
        logger.error(f"Error plotting tournament growth: {e}")
        st.error("Failed to plot tournament growth. Please try refreshing the page.")

def display_team_participation(matches_df=None) -> None:
    """
    Display team participation information using pre-computed data.
    
    Args:
        matches_df: Optional matches dataframe (not used, kept for backward compatibility)
    """
    try:
        st.subheader("Team Participation in IPL")
        
        # Load pre-computed team participation data
        overview_data = load_overview_data()
        team_df = overview_data['team_participation']
        
        if team_df.empty:
            st.warning("Team participation data is not available. Please check the data files.")
            return
        
        st.dataframe(
            team_df.sort_values('Total Matches', ascending=False),
            use_container_width=True,
            hide_index=True
        )
    except Exception as e:
        logger.error(f"Error displaying team participation: {e}")
        st.error("Failed to display team participation. Please try refreshing the page.")

def display_key_questions() -> None:
    """
    Display key questions to be explored.
    """
    try:
        st.subheader("Key Questions We'll Explore")
        
        questions = [
            "🏆 Has the dominance of certain teams changed over the course of IPL history?",
            "🎯 How significant is the toss in determining the match outcome?",
            "🌟 Which players consistently perform well in high-pressure situations?",
            "🏟️ Do certain venues favor batting or bowling teams?",
            "📊 How has the scoring pattern evolved over different seasons?"
        ]
        
        for q in questions:
            st.markdown(f"- {q}")
    except Exception as e:
        logger.error(f"Error displaying key questions: {e}")
        st.error("Failed to display key questions. Please try refreshing the page.")

def display_data_limitations() -> None:
    """
    Display data limitations.
    """
    try:
        st.info("""
        **Note on Data Limitations:**
        - Fielder information might not be consistently available for all seasons
        - Weather conditions and pitch reports are not part of the dataset
        - Detailed player statistics (age, experience, etc.) are not included
        """)
        
        # Add CricSheet attribution separately with HTML
        st.markdown("""
        **Data Source:** <a href="https://cricsheet.org/" target="_blank">CricSheet</a>
        """, unsafe_allow_html=True)
    except Exception as e:
        logger.error(f"Error displaying data limitations: {e}")

def display_dataset_info() -> None:
    """
    Display information about the dataset structure.
    """
    try:
        st.subheader("Dataset Structure")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Match-Level Information")
            st.markdown("""
            The matches dataset provides high-level information about each game:
            
            **Game Basics:**
            - `match_id`: Unique identifier for each match
            - `season`: IPL season year
            - `date`: Match date
            - `match_number`: Game sequence in the tournament
            
            **Team Information:**
            - `team1` & `team2`: Participating teams
            - `winner`: Match winner
            - `win_by_runs`/`win_by_wickets`: Victory margin
            
            **Match Conditions:**
            - `venue` & `city`: Location details
            - `toss_winner` & `toss_decision`: Toss information
            - `player_of_match`: Outstanding performer
            """)
        
        with col2:
            st.markdown("### Ball-by-Ball Details")
            st.markdown("""
            The deliveries dataset captures every ball bowled:
            
            **Batting Details:**
            - `batter` & `non_striker`: Active batsmen
            - `batsman_runs`: Runs scored by batter
            - `extra_runs`: Additional runs (wides, no-balls, etc.)
            - `total_runs`: Total runs from the delivery
            - `batter_position`: Batting order position
            
            **Bowling Information:**
            - `bowler`: Bowler name
            - `is_wicket`: Whether a wicket fell
            - `dismissal_kind`: How the batter was dismissed
            - `player_dismissed`: Dismissed player's name
            - `over` & `ball`: Delivery sequence
            """)
    except Exception as e:
        logger.error(f"Error displaying dataset info: {e}")
        st.error("Failed to display dataset information. Please try refreshing the page.") 