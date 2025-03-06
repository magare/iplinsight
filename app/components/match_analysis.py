'''\
Module for analyzing match data.
This module loads precomputed match result statistics, toss statistics, and scoring statistics,
and provides functions to display these analyses using Streamlit and Plotly.\
'''

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import os
from utils.chart_utils import responsive_plotly_chart
from pathlib import Path

# Import the dream team analysis for match details
from components.dream_team_analysis import DreamTeamAnalysis


# ---------------------------
# Helper Functions
# ---------------------------

def load_precomputed_data(filename):
    """
    Load precomputed data from Parquet files.
    
    Args:
        filename (str): Name of the Parquet file to load.
        
    Returns:
        pd.DataFrame: DataFrame containing the loaded data.
    """
    data_dir = Path(__file__).resolve().parent.parent / "data"
    
    # Convert legacy filename to parquet if needed
    if filename.endswith('.csv'):
        filename = filename.replace('.csv', '.parquet')
    
    file_path = data_dir / filename
    
    if not os.path.exists(file_path):
        st.error(f"Precomputed data file not found: {filename}")
        return pd.DataFrame()
    
    # All parquet files can be loaded with a single method, no need to handle index separately
    # as parquet preserves the index information
    return pd.read_parquet(file_path)


# ---------------------------
# Match Result Analysis Functions
# ---------------------------

def load_match_result_stats():
    """
    Load precomputed match result statistics.
    
    Returns:
        tuple: Tuple containing DataFrames with match result statistics.
    """
    # Load precomputed match result data
    runs_victories = load_precomputed_data('match_runs_victories.parquet')
    wickets_victories = load_precomputed_data('match_wickets_victories.parquet')
    win_method_by_season = load_precomputed_data('match_win_method_by_season.parquet')
    win_method_by_season_pct = load_precomputed_data('match_win_method_by_season_pct.parquet')
    
    return runs_victories, wickets_victories, win_method_by_season, win_method_by_season_pct


def display_match_result_analysis(matches_df):
    """
    Display comprehensive match result analysis using Streamlit and Plotly.

    Args:
        matches_df (pd.DataFrame): DataFrame containing match results data (not used with precomputed data).
    """
    st.subheader("Match Result Analysis")
    result_stats = load_match_result_stats()  # Load precomputed match result stats
    
    tabs = st.tabs([
        "Victory Margins",
        "Win Methods by Season",
        "Super Over Analysis"
    ])
    
    # Victory Margins Tab
    with tabs[0]:
        col1, col2 = st.columns(2)
        
        with col1:
            # Histogram for runs victories
            fig = px.histogram(
                result_stats[0],
                x='result_margin',
                title='Distribution of Victory Margins (Runs)',
                labels={'result_margin': 'Margin (Runs)'},
                nbins=20,
                template='plotly_dark',
                color_discrete_sequence=['#00ff88']
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                yaxis=dict(gridcolor='rgba(128,128,128,0.1)'),
                xaxis=dict(gridcolor='rgba(128,128,128,0.1)')
            )
            responsive_plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Histogram for wickets victories
            fig = px.histogram(
                result_stats[1],
                x='result_margin',
                title='Distribution of Victory Margins (Wickets)',
                labels={'result_margin': 'Margin (Wickets)'},
                nbins=10,
                template='plotly_dark',
                color_discrete_sequence=['#ff0088']
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                yaxis=dict(gridcolor='rgba(128,128,128,0.1)'),
                xaxis=dict(gridcolor='rgba(128,128,128,0.1)')
            )
            responsive_plotly_chart(fig, use_container_width=True)
    
    # Win Methods by Season Tab
    with tabs[1]:
        fig = px.bar(
            result_stats[2],
            title='Win Methods by Season (%)',
            barmode='stack',
            template='plotly_dark',
            color_discrete_sequence=['#00ff88', '#ff0088', '#00ffff']
        )
        fig.update_layout(
            yaxis_title='Percentage',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            yaxis=dict(gridcolor='rgba(128,128,128,0.1)'),
            xaxis=dict(gridcolor='rgba(128,128,128,0.1)')
        )
        responsive_plotly_chart(fig, use_container_width=True)
    
    # Super Over Analysis Tab
    with tabs[2]:
        st.info("Super Over analysis is not available in the current dataset")


# ---------------------------
# Toss Analysis Functions
# ---------------------------

def load_toss_stats():
    """
    Load precomputed toss statistics.
    
    Returns:
        tuple: Tuple containing DataFrames with toss statistics.
    """
    # Load precomputed toss data
    toss_decisions = load_precomputed_data('match_toss_decisions.parquet')
    toss_by_season = load_precomputed_data('match_toss_by_season.parquet')
    toss_by_season_pct = load_precomputed_data('match_toss_by_season_pct.parquet')
    toss_by_venue = load_precomputed_data('match_toss_by_venue.parquet')
    toss_by_venue_pct = load_precomputed_data('match_toss_by_venue_pct.parquet')
    toss_win_pct = load_precomputed_data('match_toss_win_pct.parquet')
    toss_decision_outcomes = load_precomputed_data('match_toss_decision_outcomes.parquet')
    
    return toss_decisions, toss_by_season, toss_by_season_pct, toss_by_venue, toss_by_venue_pct, toss_win_pct, toss_decision_outcomes


def display_toss_analysis(matches_df):
    """
    Display comprehensive toss analysis using Streamlit and Plotly.

    Args:
        matches_df (pd.DataFrame): DataFrame containing match data (not used with precomputed data).
    """
    st.subheader("Toss Analysis")
    toss_stats = load_toss_stats()
    
    tabs = st.tabs([
        "Overall Toss Impact",
        "Toss Decisions",
        "Toss by Venue"
    ])
    
    # Overall Toss Impact
    with tabs[0]:
        col1, col2 = st.columns(2)
        
        with col1:
            # Fix the TypeError by ensuring we're working with a numeric value
            toss_win_percentage = toss_stats[5]
            # Check if it's a DataFrame and extract the value if needed
            if hasattr(toss_win_percentage, 'iloc'):
                toss_win_percentage = float(toss_win_percentage.iloc[0])
            # Ensure we have a float to format
            toss_win_percentage = float(toss_win_percentage)
            st.metric("Match Win % When Winning Toss", f"{toss_win_percentage:.1f}%")
            fig = px.bar(
                toss_stats[6].reset_index(),
                x='index',
                y='win_percentage',
                title='Win % by Toss Decision',
                labels={'win_percentage': 'Win Percentage', 'index': 'Toss Decision'},
                template='plotly_dark',
                color_discrete_sequence=['#00ff88']
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                yaxis=dict(gridcolor='rgba(128,128,128,0.1)'),
                xaxis=dict(gridcolor='rgba(128,128,128,0.1)')
            )
            responsive_plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Since we can't compute correlation from precomputed data, we'll show a different metric
            try:
                # Get the win percentage difference between batting and fielding first
                win_pct_diff = abs(toss_stats[6]['win_percentage'].diff().iloc[-1])
                # Ensure it's a float for formatting
                win_pct_diff = float(win_pct_diff)
                st.metric("Toss Decision Impact", f"{win_pct_diff:.1f}%")
            except (TypeError, IndexError, AttributeError):
                # Fallback in case of any error
                st.metric("Toss Decision Impact", "N/A")
            st.caption("Difference in win % between batting and fielding first")
    
    # Toss Decisions Tab
    with tabs[1]:
        col1, col2 = st.columns(2)
        
        with col1:
            # Create a DataFrame for the pie chart
            toss_pie_data = pd.DataFrame({
                'decision': toss_stats[0].index,
                'percentage': toss_stats[0].values
            })
            
            fig = px.pie(
                toss_pie_data,
                values='percentage',
                names='decision',
                title='Toss Decision Distribution',
                template='plotly_dark',
                color_discrete_sequence=['#00ff88', '#ff0088']
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            responsive_plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.line(
                toss_stats[1],
                title='Toss Decisions by Season (%)',
                labels={'value': 'Percentage'},
                template='plotly_dark',
                color_discrete_sequence=['#00ff88', '#ff0088']
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                yaxis=dict(gridcolor='rgba(128,128,128,0.1)'),
                xaxis=dict(gridcolor='rgba(128,128,128,0.1)')
            )
            responsive_plotly_chart(fig, use_container_width=True)
    
    # Toss by Venue Tab
    with tabs[2]:
        st.subheader("Toss Decisions by Venue")
        toss_by_venue_pct = toss_stats[3]
        df_venue = toss_by_venue_pct.reset_index()
        df_melt = df_venue.melt(id_vars='venue', var_name='Toss Decision', value_name='Percentage')
        fig = px.bar(
            df_melt,
            x='venue',
            y='Percentage',
            color='Toss Decision',
            title='Toss Decision by Venue (%)',
            barmode='stack',
            template='plotly_dark',
            color_discrete_sequence=['#00ff88', '#ff0088']
        )
        fig.update_layout(
            xaxis_title='Venue',
            yaxis_title='Percentage',
            xaxis_tickangle=-45,
            margin=dict(t=60, b=80),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            yaxis=dict(gridcolor='rgba(128,128,128,0.1)'),
            xaxis=dict(gridcolor='rgba(128,128,128,0.1)')
        )
        responsive_plotly_chart(fig, use_container_width=True)


# ---------------------------
# Scoring Analysis Functions
# ---------------------------

def load_scoring_stats():
    """
    Load precomputed scoring statistics.
    
    Returns:
        tuple: Tuple containing DataFrames with scoring statistics.
    """
    # Load precomputed scoring data
    match_scores = load_precomputed_data('match_scores.parquet')
    venue_scores = load_precomputed_data('match_venue_scores.parquet')
    season_scores = load_precomputed_data('match_season_scores.parquet')
    phase_avg = load_precomputed_data('match_phase_avg.parquet')
    
    return match_scores, venue_scores, season_scores, phase_avg


def load_high_low_scoring_stats():
    """
    Load precomputed high and low scoring statistics.
    
    Returns:
        tuple: Tuple containing DataFrames with high and low scoring statistics.
    """
    # Load precomputed high/low scoring data
    score_thresholds = load_precomputed_data('match_score_thresholds.parquet')
    high_scoring_venues = load_precomputed_data('match_high_scoring_venues.parquet')
    low_scoring_venues = load_precomputed_data('match_low_scoring_venues.parquet')
    high_scoring_teams = load_precomputed_data('match_high_scoring_teams.parquet')
    low_scoring_teams = load_precomputed_data('match_low_scoring_teams.parquet')
    high_scoring_seasons = load_precomputed_data('match_high_scoring_seasons.parquet')
    low_scoring_seasons = load_precomputed_data('match_low_scoring_seasons.parquet')
    high_scoring_toss = load_precomputed_data('match_high_scoring_toss.parquet')
    low_scoring_toss = load_precomputed_data('match_low_scoring_toss.parquet')
    high_scoring_phases = load_precomputed_data('match_high_scoring_phases.parquet')
    low_scoring_phases = load_precomputed_data('match_low_scoring_phases.parquet')
    
    return (score_thresholds, high_scoring_venues, low_scoring_venues, high_scoring_teams, 
            low_scoring_teams, high_scoring_seasons, low_scoring_seasons, high_scoring_toss, 
            low_scoring_toss, high_scoring_phases, low_scoring_phases)


def display_scoring_analysis(matches_df, deliveries_df):
    """
    Display comprehensive scoring analysis using Streamlit and Plotly.

    Args:
        matches_df (pd.DataFrame): DataFrame containing match results data.
        deliveries_df (pd.DataFrame): DataFrame containing ball-by-ball data.
    """
    st.subheader("Scoring Patterns Analysis")
    scoring_stats = load_scoring_stats()
    
    tabs = st.tabs([
        "Venue Analysis",
        "Season Trends",
        "Phase Analysis",
        "Score Distribution"
    ])
    
    # Venue Analysis Tab
    with tabs[0]:
        st.subheader("Scoring by Venue")
        venue_scores_df = scoring_stats[1].reset_index()
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name='First Innings',
            x=venue_scores_df['venue'],
            y=venue_scores_df['first_innings_avg'],
            marker_color='#00ff88'
        ))
        fig.add_trace(go.Bar(
            name='Second Innings',
            x=venue_scores_df['venue'],
            y=venue_scores_df['second_innings_avg'],
            marker_color='#ff0088'
        ))
        fig.update_layout(
            title='Average Scores by Venue',
            barmode='group',
            xaxis_tickangle=-45,
            height=500,
            template='plotly_dark',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            yaxis=dict(gridcolor='rgba(128,128,128,0.1)'),
            xaxis=dict(gridcolor='rgba(128,128,128,0.1)')
        )
        responsive_plotly_chart(fig, use_container_width=True)
    
    # Season Trends Tab
    with tabs[1]:
        season_scores_df = scoring_stats[2].reset_index()
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            name='First Innings',
            x=season_scores_df['season'],
            y=season_scores_df['first_innings_avg'],
            mode='lines+markers',
            line=dict(color='#00ff88'),
            error_y=dict(
                type='data',
                array=season_scores_df['first_innings_std'],
                visible=True,
                color='#00ff88'
            )
        ))
        fig.add_trace(go.Scatter(
            name='Second Innings',
            x=season_scores_df['season'],
            y=season_scores_df['second_innings_avg'],
            mode='lines+markers',
            line=dict(color='#ff0088'),
            error_y=dict(
                type='data',
                array=season_scores_df['second_innings_std'],
                visible=True,
                color='#ff0088'
            )
        ))
        fig.update_layout(
            title='Scoring Trends by Season',
            xaxis_title='Season',
            yaxis_title='Average Score',
            height=500,
            template='plotly_dark',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            yaxis=dict(gridcolor='rgba(128,128,128,0.1)'),
            xaxis=dict(gridcolor='rgba(128,128,128,0.1)')
        )
        responsive_plotly_chart(fig, use_container_width=True)
    
    # Phase Analysis Tab
    with tabs[2]:
        phase_scores_df = scoring_stats[3].reset_index()
        fig = px.bar(
            phase_scores_df,
            x='phase',
            y='mean',
            color='inning',
            barmode='group',
            title='Average Runs by Phase',
            error_y='std',
            labels={'mean': 'Average Runs', 'inning': 'Innings'},
            template='plotly_dark',
            color_discrete_sequence=['#00ff88', '#ff0088']
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            yaxis=dict(gridcolor='rgba(128,128,128,0.1)'),
            xaxis=dict(gridcolor='rgba(128,128,128,0.1)')
        )
        responsive_plotly_chart(fig, use_container_width=True)
    
    # Score Distribution Tab
    with tabs[3]:
        col1, col2 = st.columns(2)
        with col1:
            fig = px.box(
                scoring_stats[0],
                y=['first_innings', 'second_innings'],
                title='Score Distribution by Innings',
                labels={'value': 'Score', 'variable': 'Innings'},
                template='plotly_dark',
                color_discrete_sequence=['#00ff88', '#ff0088']
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                yaxis=dict(gridcolor='rgba(128,128,128,0.1)'),
                xaxis=dict(gridcolor='rgba(128,128,128,0.1)')
            )
            responsive_plotly_chart(fig, use_container_width=True)
        with col2:
            st.subheader("Summary Statistics")
            summary_df = pd.DataFrame({
                'First Innings': scoring_stats[0]['first_innings'].describe(),
                'Second Innings': scoring_stats[0]['second_innings'].describe()
            }).round(2)
            st.dataframe(summary_df)


def display_high_low_scoring_analysis(matches_df, deliveries_df):
    """
    Display analysis of high and low scoring matches using Streamlit and Plotly.

    Args:
        matches_df (pd.DataFrame): DataFrame containing match data (not used with precomputed data).
        deliveries_df (pd.DataFrame): DataFrame containing delivery data (not used with precomputed data).
    """
    st.subheader("High/Low Scoring Matches Analysis")
    scoring_stats = load_high_low_scoring_stats()
    
    col1, col2 = st.columns(2)
    with col1:
        try:
            high_threshold = float(scoring_stats[0]['high_threshold'].iloc[0])
            st.metric("High Scoring Threshold", f"{high_threshold:.0f} runs")
        except (AttributeError, KeyError, IndexError, ValueError, TypeError):
            # Try alternative access methods if the normal one fails
            try:
                if isinstance(scoring_stats[0], dict) and 'high_threshold' in scoring_stats[0]:
                    high_threshold = float(scoring_stats[0]['high_threshold'])
                    st.metric("High Scoring Threshold", f"{high_threshold:.0f} runs")
                else:
                    st.metric("High Scoring Threshold", "N/A")
            except:
                st.metric("High Scoring Threshold", "N/A")
    with col2:
        try:
            low_threshold = float(scoring_stats[0]['low_threshold'].iloc[0])
            st.metric("Low Scoring Threshold", f"{low_threshold:.0f} runs")
        except (AttributeError, KeyError, IndexError, ValueError, TypeError):
            # Try alternative access methods if the normal one fails
            try:
                if isinstance(scoring_stats[0], dict) and 'low_threshold' in scoring_stats[0]:
                    low_threshold = float(scoring_stats[0]['low_threshold'])
                    st.metric("Low Scoring Threshold", f"{low_threshold:.0f} runs")
                else:
                    st.metric("Low Scoring Threshold", "N/A")
            except:
                st.metric("Low Scoring Threshold", "N/A")
    
    tabs = st.tabs([
        "Venue Analysis",
        "Team Analysis",
        "Season Analysis",
        "Match Characteristics"
    ])
    
    # Venue Analysis Tab
    with tabs[0]:
        col1, col2 = st.columns(2)
        with col1:
            fig = px.bar(
                scoring_stats[1].head(10),
                title='Top Venues for High Scoring Matches',
                labels={'value': 'Number of Matches', 'index': 'Venue'}
            )
            fig.update_layout(xaxis_tickangle=-45)
            responsive_plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.bar(
                scoring_stats[2].head(10),
                title='Top Venues for Low Scoring Matches',
                labels={'value': 'Number of Matches', 'index': 'Venue'}
            )
            fig.update_layout(xaxis_tickangle=-45)
            responsive_plotly_chart(fig, use_container_width=True)
    
    # Team Analysis Tab
    with tabs[1]:
        col1, col2 = st.columns(2)
        with col1:
            fig = px.bar(
                scoring_stats[3].head(10),
                title='Teams in High Scoring Matches',
                labels={'value': 'Number of Matches', 'index': 'Team'}
            )
            fig.update_layout(xaxis_tickangle=-45)
            responsive_plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.bar(
                scoring_stats[4].head(10),
                title='Teams in Low Scoring Matches',
                labels={'value': 'Number of Matches', 'index': 'Team'}
            )
            fig.update_layout(xaxis_tickangle=-45)
            responsive_plotly_chart(fig, use_container_width=True)
    
    # Season Analysis Tab
    with tabs[2]:
        col1, col2 = st.columns(2)
        with col1:
            fig = px.bar(
                scoring_stats[5],
                title='High Scoring Matches by Season',
                labels={'value': 'Number of Matches', 'index': 'Season'}
            )
            fig.update_layout(xaxis_tickangle=-45)
            responsive_plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.bar(
                scoring_stats[6],
                title='Low Scoring Matches by Season',
                labels={'value': 'Number of Matches', 'index': 'Season'}
            )
            fig.update_layout(xaxis_tickangle=-45)
            responsive_plotly_chart(fig, use_container_width=True)
    
    # Match Characteristics Tab
    with tabs[3]:
        col1, col2 = st.columns(2)
        with col1:
            fig = go.Figure(data=[
                go.Bar(name='High Scoring', x=scoring_stats[7].index, 
                       y=scoring_stats[7].values),
                go.Bar(name='Low Scoring', x=scoring_stats[8].index, 
                       y=scoring_stats[8].values)
            ])
            fig.update_layout(
                title='Toss Decisions in High/Low Scoring Matches',
                barmode='group'
            )
            responsive_plotly_chart(fig, use_container_width=True)
        with col2:
            # Fix the DataFrame creation by checking structure and providing an index if needed
            try:
                # Check if the data has indexes
                if hasattr(scoring_stats[9], 'index') and hasattr(scoring_stats[10], 'index'):
                    # If both have matching indexes, use them
                    phase_comparison = pd.DataFrame({
                        'High Scoring': scoring_stats[9],
                        'Low Scoring': scoring_stats[10]
                    })
                else:
                    # If they're scalar values or don't have indexes, create an index
                    # First check if they are dictionaries
                    if isinstance(scoring_stats[9], dict) and isinstance(scoring_stats[10], dict):
                        # Use the dictionary keys as index
                        phase_comparison = pd.DataFrame({
                            'High Scoring': list(scoring_stats[9].values()),
                            'Low Scoring': list(scoring_stats[10].values())
                        }, index=list(scoring_stats[9].keys()))
                    else:
                        # Try to convert to series if they're not already
                        high_scoring = pd.Series(scoring_stats[9]) if not isinstance(scoring_stats[9], pd.Series) else scoring_stats[9]
                        low_scoring = pd.Series(scoring_stats[10]) if not isinstance(scoring_stats[10], pd.Series) else scoring_stats[10]
                        
                        # Create a DataFrame with an explicit index
                        if len(high_scoring) == 3 and len(low_scoring) == 3:
                            # Assume it's powerplay, middle, death overs
                            phase_comparison = pd.DataFrame({
                                'High Scoring': high_scoring.values,
                                'Low Scoring': low_scoring.values
                            }, index=['Powerplay', 'Middle Overs', 'Death Overs'])
                        else:
                            # Use numeric indices
                            phase_comparison = pd.DataFrame({
                                'High Scoring': high_scoring.values,
                                'Low Scoring': low_scoring.values
                            }, index=[f"Phase {i+1}" for i in range(len(high_scoring))])
                
                fig = px.bar(
                    phase_comparison,
                    barmode='group',
                    title='Run Rates by Phase',
                    labels={'value': 'Average Runs', 'variable': 'Match Type'}
                )
                responsive_plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error displaying phase comparison: {e}")
                st.info("The phase comparison data could not be displayed due to a formatting issue.")


def display_match_dream_team(match_id):
    """
    Display the dream team for a specific match.
    
    Args:
        match_id (str): The ID of the match to display the dream team for.
    """
    try:
        dt_instance = DreamTeamAnalysis()
        dream_team_df = dt_instance.get_match_dream_team(str(match_id))
        
        if not dream_team_df.empty:
            st.subheader("Match Dream Team")
            
            # Display dream team data in a nicely formatted table
            st.dataframe(
                dream_team_df[[
                    'player', 'role', 'captain_role', 'total_points',
                    'batting_points', 'bowling_points', 'fielding_points'
                ]].style.format({
                    'total_points': '{:.2f}',
                    'batting_points': '{:.2f}',
                    'bowling_points': '{:.2f}',
                    'fielding_points': '{:.2f}'
                }).background_gradient(subset=['total_points'], cmap='YlOrRd'),
                hide_index=True,
                use_container_width=True
            )
            
            # Display points breakdown
            col1, col2 = st.columns(2)
            with col1:
                # Performance breakdown chart
                fig = px.bar(
                    dream_team_df,
                    x='player',
                    y=['batting_points', 'bowling_points', 'fielding_points'],
                    title='Player Points Breakdown',
                    labels={'value': 'Points', 'variable': 'Category'},
                    barmode='stack'
                )
                fig.update_layout(height=400)
                responsive_plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Role distribution
                if 'role' in dream_team_df.columns:
                    role_counts = dream_team_df['role'].value_counts()
                    fig = px.pie(
                        values=role_counts.values,
                        names=role_counts.index,
                        title='Role Distribution in Dream Team'
                    )
                    fig.update_layout(height=400)
                    responsive_plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Dream team data not available for this match")
    except Exception as e:
        st.error(f"Error displaying dream team: {str(e)}")

# Add a new function to display individual match details
def display_match_details(matches_df, match_id=None):
    """
    Display details for a specific match, including participants, outcome, and dream team.
    
    Args:
        matches_df (pd.DataFrame): DataFrame containing match data.
        match_id (int): Optional specific match ID to display.
    """
    if match_id is None:
        match_id = st.selectbox("Select Match", 
                               options=sorted(matches_df['match_id'].unique()),
                               format_func=lambda x: f"Match {x}: {matches_df[matches_df['match_id'] == x]['team1'].values[0]} vs {matches_df[matches_df['match_id'] == x]['team2'].values[0]} ({matches_df[matches_df['match_id'] == x]['date'].dt.date.values[0] if pd.api.types.is_datetime64_any_dtype(matches_df['date']) else matches_df[matches_df['match_id'] == x]['date'].values[0].split('T')[0] if isinstance(matches_df[matches_df['match_id'] == x]['date'].values[0], str) and 'T' in matches_df[matches_df['match_id'] == x]['date'].values[0] else matches_df[matches_df['match_id'] == x]['date'].values[0]})")
    
    match_data = matches_df[matches_df['match_id'] == match_id].iloc[0]
    
    # Format the date - remove timestamp if present
    formatted_date = match_data['date']
    if pd.api.types.is_datetime64_any_dtype(matches_df['date']):
        formatted_date = match_data['date'].date()
    elif isinstance(match_data['date'], str):
        if 'T' in match_data['date']:
            formatted_date = match_data['date'].split('T')[0]
        elif ' ' in match_data['date'] and ':' in match_data['date']:
            formatted_date = match_data['date'].split(' ')[0]
    
    # Match header
    st.header(f"{match_data['team1']} vs {match_data['team2']} - {formatted_date} - {match_data['venue']}")
    
    # Match Details
    st.subheader("Match Details:")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        - **Winner:** {match_data['winner']}
        - **Toss Winner:** {match_data['toss_winner']} ({match_data['toss_decision']})
        - **Player of Match:** {match_data['player_of_match']}
        """)
    with col2:
        st.markdown(f"""
        - **Venue:** {match_data['venue']}
        - **Season:** {match_data['season']}
        - **City:** {match_data['city'] if 'city' in match_data else 'N/A'}
        """)
    
    # Display dream team for the match
    display_match_dream_team(match_id) 