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
from typing import Dict, Any, Optional, Tuple, List

# Add parent directory to path to allow importing from config and utils
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import PRECOMPUTED_DATA, MAX_ROWS_DISPLAY
from utils.chart_utils import responsive_plotly_chart, get_neon_color_discrete_sequence
from utils.data_loader import load_precomputed_data, format_large_number, calculate_basic_stats
from utils.color_palette import NEON_COLORS, CHART_STYLING
from utils.error_handler import error_boundary, ErrorBoundary
from utils.state_manager import get_state, is_mobile
from utils.ui_components import (
    HeaderComponent, 
    DataDisplayComponent,
    LayoutComponent
)

# Configure logging
logger = logging.getLogger(__name__)

@error_boundary
def render_overview_section(matches: pd.DataFrame, deliveries: pd.DataFrame) -> None:
    """
    Render the overview section.
    
    Args:
        matches: Matches dataframe
        deliveries: Deliveries dataframe
    """
    # Display app header
    HeaderComponent.main_header()
    
    # Display section heading
    HeaderComponent.section_header(
        "IPL Data Analysis Overview",
        "The Indian Premier League (IPL) has captivated cricket fans worldwide since its inception in 2008. "
        "This analysis dives deep into the rich history of IPL, uncovering patterns and insights from over a decade of matches."
    )
    
    # Display metrics dashboard
    display_overview_metrics(matches, deliveries)
    
    # Show tournament growth
    with ErrorBoundary("Tournament Growth Section"):
        HeaderComponent.subsection_header("Tournament Growth Over Years")
        plot_tournament_growth(matches, deliveries)
    
    # Show team participation table
    with ErrorBoundary("Team Participation Section"):
        display_team_participation(matches)
    
    # Display other informational sections
    with ErrorBoundary("Informational Sections"):
        display_key_questions()
        display_data_limitations()
        display_dataset_info()

@error_boundary
def display_overview_metrics(matches: pd.DataFrame, deliveries: pd.DataFrame) -> None:
    """
    Display overview metrics dashboard.
    
    Args:
        matches: Matches dataframe
        deliveries: Deliveries dataframe
    """
    # Calculate basic stats
    stats = calculate_basic_stats(matches, deliveries)
    
    # Group metrics by category
    tournament_metrics = {
        "Total Matches": format_large_number(stats.get('total_matches', 0)),
        "Unique Teams": format_large_number(stats.get('total_teams', 0)),
        "Seasons": format_large_number(stats.get('total_seasons', 0)),
        "Venues": format_large_number(stats.get('total_venues', 0)),
        "Cities": format_large_number(stats.get('total_cities', 0))
    }
    
    batting_metrics = {
        "Total Runs": format_large_number(stats.get('total_runs', 0)),
        "Total Boundaries": format_large_number(stats.get('total_boundaries', 0)),
        "Total Sixes": format_large_number(stats.get('total_sixes', 0)),
        "Avg. Runs/Match": f"{float(stats.get('avg_runs_per_match', 0)):.1f}" if 'avg_runs_per_match' in stats else "N/A"
    }
    
    bowling_metrics = {
        "Total Wickets": format_large_number(stats.get('total_wickets', 0)),
        "Total Overs": format_large_number(int(stats.get('total_overs', 0))) if 'total_overs' in stats else "N/A",
        "Avg. Wickets/Match": f"{float(stats.get('avg_wickets_per_match', 0)):.1f}" if 'avg_wickets_per_match' in stats else "N/A"
    }
    
    # Responsive layout
    # Get device type to determine columns
    device_type = get_state('device_type', 'desktop')
    
    if device_type == 'mobile':
        # For mobile, show metrics in a single column
        HeaderComponent.subsection_header("Tournament Scale")
        DataDisplayComponent.metrics_display(tournament_metrics, 1)
        
        HeaderComponent.subsection_header("Batting Insights")
        DataDisplayComponent.metrics_display(batting_metrics, 1)
        
        HeaderComponent.subsection_header("Bowling Insights")
        DataDisplayComponent.metrics_display(bowling_metrics, 1)
    else:
        # For desktop, show metrics in three columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            HeaderComponent.subsection_header("Tournament Scale")
            DataDisplayComponent.metrics_display(tournament_metrics, 1)
            
        with col2:
            HeaderComponent.subsection_header("Batting Insights")
            DataDisplayComponent.metrics_display(batting_metrics, 1)
            
        with col3:
            HeaderComponent.subsection_header("Bowling Insights")
            DataDisplayComponent.metrics_display(bowling_metrics, 1)

@error_boundary
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
        return {}

@error_boundary
def plot_tournament_growth(matches_df=None, deliveries_df=None) -> None:
    """
    Plot tournament growth over the years.
    
    Args:
        matches_df: Matches dataframe
        deliveries_df: Deliveries dataframe
    """
    # Load precomputed data if needed
    data = load_overview_data()
    
    # Set up the columns based on device type
    device_type = get_state('device_type', 'desktop')
    if device_type == 'mobile':
        # For mobile, stack plots vertically
        col1 = st.container()
        col2 = st.container()
        col3 = st.container()
    else:
        # For desktop, show side by side
        col1, col2, col3 = st.columns(3)
    
    with col1:
        # Matches per season
        if 'matches_per_season' in data and not data['matches_per_season'].empty:
            df = data['matches_per_season']
            fig = px.bar(
                df, 
                x='season', 
                y='matches',
                title='Matches per Season',
                labels={'season': 'Season', 'matches': 'Number of Matches'},
                color_discrete_sequence=get_neon_color_discrete_sequence(1)
            )
            # Apply common styling
            fig.update_layout(**CHART_STYLING)
            responsive_plotly_chart(fig)
        else:
            # Compute on the fly if precomputed data is not available
            if matches_df is not None:
                try:
                    matches_per_season = matches_df.groupby('season').size().reset_index(name='matches')
                    fig = px.bar(
                        matches_per_season, 
                        x='season', 
                        y='matches',
                        title='Matches per Season',
                        labels={'season': 'Season', 'matches': 'Number of Matches'},
                        color_discrete_sequence=get_neon_color_discrete_sequence(1)
                    )
                    fig.update_layout(**CHART_STYLING)
                    responsive_plotly_chart(fig)
                except Exception as e:
                    logger.error(f"Error generating matches per season chart: {e}")
                    st.error("Could not generate matches per season chart.")
    
    with col2:
        # Average runs per season
        if 'avg_runs_by_season' in data and not data['avg_runs_by_season'].empty:
            df = data['avg_runs_by_season']
            fig = px.line(
                df, 
                x='season', 
                y='avg_runs',
                title='Average Runs per Match by Season',
                labels={'season': 'Season', 'avg_runs': 'Average Runs per Match'},
                markers=True,
                color_discrete_sequence=get_neon_color_discrete_sequence(1, 1)
            )
            fig.update_layout(**CHART_STYLING)
            responsive_plotly_chart(fig)
        else:
            # Compute on the fly if precomputed data is not available
            if matches_df is not None and deliveries_df is not None:
                try:
                    # Join the deliveries and matches dataframes
                    runs_by_match = deliveries_df.groupby('match_id')['total_runs'].sum().reset_index()
                    matches_with_runs = pd.merge(
                        matches_df[['id', 'season']], 
                        runs_by_match, 
                        left_on='id', 
                        right_on='match_id',
                        how='inner'
                    )
                    
                    # Calculate average runs per season
                    avg_runs_by_season = matches_with_runs.groupby('season')['total_runs'].mean().reset_index()
                    avg_runs_by_season = avg_runs_by_season.rename(columns={'total_runs': 'avg_runs'})
                    
                    fig = px.line(
                        avg_runs_by_season, 
                        x='season', 
                        y='avg_runs',
                        title='Average Runs per Match by Season',
                        labels={'season': 'Season', 'avg_runs': 'Average Runs per Match'},
                        markers=True,
                        color_discrete_sequence=get_neon_color_discrete_sequence(1, 1)
                    )
                    fig.update_layout(**CHART_STYLING)
                    responsive_plotly_chart(fig)
                except Exception as e:
                    logger.error(f"Error generating average runs chart: {e}")
                    st.error("Could not generate average runs chart.")
    
    with col3:
        # Average wickets per season
        if 'avg_wickets_by_season' in data and not data['avg_wickets_by_season'].empty:
            df = data['avg_wickets_by_season']
            fig = px.line(
                df, 
                x='season', 
                y='avg_wickets',
                title='Average Wickets per Match by Season',
                labels={'season': 'Season', 'avg_wickets': 'Average Wickets per Match'},
                markers=True,
                color_discrete_sequence=get_neon_color_discrete_sequence(1, 2)
            )
            fig.update_layout(**CHART_STYLING)
            responsive_plotly_chart(fig)
        else:
            # Compute on the fly if precomputed data is not available
            if matches_df is not None and deliveries_df is not None:
                try:
                    # Calculate wickets by match
                    wickets_by_match = deliveries_df.groupby('match_id')['is_wicket'].sum().reset_index()
                    
                    # Join with matches to get season info
                    matches_with_wickets = pd.merge(
                        matches_df[['id', 'season']], 
                        wickets_by_match, 
                        left_on='id', 
                        right_on='match_id',
                        how='inner'
                    )
                    
                    # Calculate average wickets per season
                    avg_wickets_by_season = matches_with_wickets.groupby('season')['is_wicket'].mean().reset_index()
                    avg_wickets_by_season = avg_wickets_by_season.rename(columns={'is_wicket': 'avg_wickets'})
                    
                    fig = px.line(
                        avg_wickets_by_season, 
                        x='season', 
                        y='avg_wickets',
                        title='Average Wickets per Match by Season',
                        labels={'season': 'Season', 'avg_wickets': 'Average Wickets per Match'},
                        markers=True,
                        color_discrete_sequence=get_neon_color_discrete_sequence(1, 2)
                    )
                    fig.update_layout(**CHART_STYLING)
                    responsive_plotly_chart(fig)
                except Exception as e:
                    logger.error(f"Error generating average wickets chart: {e}")
                    st.error("Could not generate average wickets chart.")

@error_boundary
def display_team_participation(matches_df=None) -> None:
    """
    Display team participation table.
    
    Args:
        matches_df: Matches dataframe
    """
    HeaderComponent.subsection_header("Team Participation Over the Years")
    
    # Load precomputed data if needed
    data = load_overview_data()
    
    if 'team_participation' in data and not data['team_participation'].empty:
        df = data['team_participation']
        # Format win rate as percentage
        if 'Win Rate (%)' in df.columns:
            df['Win Rate (%)'] = df['Win Rate (%)'].apply(lambda x: f"{x:.1f}%")
        
        # Display the dataframe
        DataDisplayComponent.table_with_download(
            df, 
            download_label="Download Team Data", 
            filename="team_participation.csv"
        )
    else:
        # Compute on the fly if precomputed data is not available
        if matches_df is not None:
            try:
                # Extract all team names
                teams1 = matches_df['team1'].value_counts().reset_index()
                teams1.columns = ['team', 'count1']
                teams2 = matches_df['team2'].value_counts().reset_index()
                teams2.columns = ['team', 'count2']
                
                # Merge to get total matches
                teams = pd.merge(teams1, teams2, on='team', how='outer').fillna(0)
                teams['Total Matches'] = teams['count1'] + teams['count2']
                
                # Get wins
                winner_counts = matches_df['winner'].value_counts().reset_index()
                winner_counts.columns = ['team', 'Wins']
                
                # Merge with total matches
                team_stats = pd.merge(teams[['team', 'Total Matches']], winner_counts, on='team', how='left').fillna(0)
                
                # Calculate win rate
                team_stats['Win Rate (%)'] = (team_stats['Wins'] / team_stats['Total Matches'] * 100).round(1)
                team_stats['Win Rate (%)'] = team_stats['Win Rate (%)'].apply(lambda x: f"{x:.1f}%")
                
                # Get seasons played
                team_seasons = pd.concat([
                    matches_df[['team1', 'season']].rename(columns={'team1': 'team'}),
                    matches_df[['team2', 'season']].rename(columns={'team2': 'team'})
                ])
                seasons_played = team_seasons.groupby('team')['season'].nunique().reset_index()
                seasons_played.columns = ['team', 'Seasons Played']
                
                # Final merge
                team_stats = pd.merge(team_stats, seasons_played, on='team', how='left')
                team_stats = team_stats.rename(columns={'team': 'Team'})
                
                # Sort by total matches
                team_stats = team_stats.sort_values('Total Matches', ascending=False).reset_index(drop=True)
                
                # Select and order columns
                team_stats = team_stats[['Team', 'Seasons Played', 'Total Matches', 'Wins', 'Win Rate (%)']]
                
                # Display the dataframe
                DataDisplayComponent.table_with_download(
                    team_stats, 
                    download_label="Download Team Data", 
                    filename="team_participation.csv"
                )
            except Exception as e:
                logger.error(f"Error generating team participation table: {e}")
                st.error("Could not generate team participation table.")

@error_boundary
def display_key_questions() -> None:
    """
    Display key questions addressed in the analysis.
    """
    with LayoutComponent.expander("Key Questions Addressed in This Analysis", expanded=False):
        st.markdown("""
        This analysis explores the following key questions about IPL cricket:
        
        1. **Team Performance**: Which teams have been the most successful in IPL history?
        2. **Player Impact**: Who are the most valuable players in terms of batting, bowling, and all-round performance?
        3. **Match Dynamics**: How do factors like toss, venue, and batting order impact match outcomes?
        4. **Season Trends**: How has the game evolved across different IPL seasons?
        5. **Venue Analysis**: Which venues favor batting or bowling, and how do teams perform at different grounds?
        6. **Dream Team**: Based on historical data, what would an all-time IPL dream team look like?
        
        Use the navigation sidebar to explore each of these aspects in detail.
        """)

@error_boundary
def display_data_limitations() -> None:
    """
    Display data limitations.
    """
    with LayoutComponent.expander("Data Limitations", expanded=False):
        st.markdown("""
        While this analysis offers valuable insights into IPL cricket, it's important to acknowledge some limitations:
        
        - **Historical Coverage**: The dataset includes matches from 2008 to recent seasons, but may not cover the most recent matches.
        - **Missing Variables**: Some detailed aspects like field placements, player movements, and exact ball trajectories are not captured.
        - **Team Changes**: Team names and ownership have changed over the years (e.g., Delhi Daredevils → Delhi Capitals).
        - **Player Consistency**: Player names may have slight variations across the dataset.
        - **Context Factors**: External factors like weather conditions, injuries, and pitch condition details are not fully represented.
        
        The analysis focuses on the patterns and trends that can be reliably extracted from the available data.
        """)

@error_boundary
def display_dataset_info() -> None:
    """
    Display dataset structure information.
    """
    with LayoutComponent.expander("Dataset Structure", expanded=False):
        st.markdown("""
        ### IPL Dataset Structure
        
        This analysis is based on two primary datasets:
        
        #### Matches Dataset
        Contains match-level information including:
        - Match ID, date, venue, teams, toss details
        - Match results, player of the match
        - Team compositions and match officials
        
        #### Deliveries Dataset
        Contains ball-by-ball details including:
        - Ball information (over, ball number)
        - Batting and bowling players
        - Runs scored, wickets, extras
        - Dismissal details
        
        These datasets together provide a comprehensive view of IPL matches from both macro (match) and micro (ball-by-ball) perspectives.
        """)

        # Show dataset schema
        if st.checkbox("Show Dataset Schema"):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### Matches Schema")
                st.code("""
                - id: Match ID
                - season: IPL Season year
                - city: Host city
                - date: Match date
                - team1: First team
                - team2: Second team
                - toss_winner: Team winning the toss
                - toss_decision: Bat or field
                - result: Match result type
                - winner: Winning team
                - player_of_match: Player of the match
                - venue: Stadium
                - umpire1, umpire2: Match officials
                """)
            
            with col2:
                st.markdown("#### Deliveries Schema")
                st.code("""
                - match_id: Match ID
                - inning: Innings number
                - batting_team: Team batting
                - bowling_team: Team bowling
                - over: Over number
                - ball: Ball number in the over
                - batsman: Batsman facing
                - non_striker: Non-striking batsman
                - bowler: Bowler bowling
                - batsman_runs: Runs scored by batsman
                - extra_runs: Extra runs
                - total_runs: Total runs for the delivery
                - is_wicket: Whether wicket fell (0/1)
                - dismissal_kind: How batsman got out
                - player_dismissed: Dismissed player
                - fielder: Fielder involved in dismissal
                """) 