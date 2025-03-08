import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
import io
import pandas as pd
import os
from pathlib import Path
from utils.chart_utils import responsive_plotly_chart, update_chart_for_responsive_layout, get_neon_color_discrete_sequence
from utils.color_palette import NEON_COLORS, CHART_STYLING, apply_neon_style

def load_overview_data():
    """Load pre-computed data for the overview section."""
    base_path = Path(__file__).resolve().parent.parent / "data"
    
    # Load pre-computed datasets
    matches_per_season = pd.read_parquet(base_path / "overview_matches_per_season.parquet")
    avg_runs_by_season = pd.read_parquet(base_path / "overview_avg_runs_by_season.parquet")
    avg_wickets_by_season = pd.read_parquet(base_path / "overview_avg_wickets_by_season.parquet")
    team_participation = pd.read_parquet(base_path / "overview_team_participation.parquet")
    
    # Rename columns to make them more human-readable
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

def plot_tournament_growth(matches_df=None, deliveries_df=None):
    """Plot tournament growth statistics using pre-computed data."""
    # Load pre-computed data
    overview_data = load_overview_data()
    matches_per_season = overview_data['matches_per_season']
    avg_runs_by_season = overview_data['avg_runs_by_season']
    avg_wickets_by_season = overview_data['avg_wickets_by_season']
    
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
                dtick=5,  # Tick every 5 matches
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
                dtick=5,  # Tick every 5 runs
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
                dtick=0.5,  # Tick every 0.5 wickets
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

def display_team_participation(matches_df=None):
    """Display team participation information using pre-computed data."""
    st.subheader("Team Participation in IPL")
    
    # Load pre-computed team participation data
    overview_data = load_overview_data()
    team_df = overview_data['team_participation']
    
    st.dataframe(
        team_df.sort_values('Total Matches', ascending=False),
        use_container_width=True,
        hide_index=True
    )

def display_key_questions():
    """Display key questions to be explored."""
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

def display_data_limitations():
    """Display data limitations."""
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

def display_dataset_info():
    """Display information about the dataset structure."""
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
        - `is_wicket`: Wicket indicator
        - `wicket_kind`: Type of dismissal
        - `fielder`: Fielder involved (if applicable)
        
        **Game Context:**
        - `inning`: 1st or 2nd innings
        - `over` & `ball`: Delivery sequence
        """) 