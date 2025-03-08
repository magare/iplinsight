import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import numpy as np
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from utils.chart_utils import responsive_plotly_chart, update_chart_for_responsive_layout

# Define consistent color scheme to match the rest of the app
COLOR_SEQUENCE = px.colors.qualitative.Plotly
TEMPLATE = "plotly_white"

# ---------------------------
# Helper Functions
# ---------------------------

def load_precomputed_json(filename):
    """
    Load precomputed data from JSON files.
    
    Args:
        filename (str): Name of the JSON file to load.
        
    Returns:
        dict: Dictionary containing the loaded data.
    """
    data_dir = Path(__file__).resolve().parent.parent / "data"
    file_path = data_dir / filename
    
    if not os.path.exists(file_path):
        st.error(f"Precomputed data file not found: {filename}")
        return {}
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            return data
    except json.JSONDecodeError:
        st.error(f"Error parsing JSON file: {filename}. The file may be corrupted.")
        return {}
    except Exception as e:
        st.error(f"Error loading {filename}: {str(e)}")
        return {}

def load_precomputed_parquet(filename):
    """
    Load precomputed data from Parquet files.
    
    Args:
        filename (str): Name of the Parquet file to load.
        
    Returns:
        dict or DataFrame: Data from the parquet file.
    """
    data_dir = Path(__file__).resolve().parent.parent / "data"
    file_path = data_dir / filename
    
    if not os.path.exists(file_path):
        st.warning(f"Precomputed data file not found: {filename}")
        return {}
    
    try:
        df = pd.read_parquet(file_path)
        return df
    except Exception as e:
        st.error(f"Error loading {filename}: {str(e)}")
        return {}

def load_venue_metadata():
    """Load precomputed venue metadata."""
    return load_precomputed_parquet('venue_metadata.json')

def load_venue_team_performance():
    """Load precomputed venue team performance data."""
    return load_precomputed_parquet('venue_team_performance.json')

def load_venue_scoring_patterns():
    """Load precomputed venue scoring patterns data."""
    return load_precomputed_parquet('venue_scoring_patterns.json')

def load_venue_toss_impact():
    """Load precomputed venue toss impact data."""
    return load_precomputed_parquet('venue_toss_impact.json')

def load_venue_weather_impact():
    """Load precomputed venue weather impact data."""
    return load_precomputed_parquet('venue_weather_impact.json')

def load_venue_team_stats():
    """Load precomputed venue team stats data."""
    return load_precomputed_parquet('venue_team_stats.parquet')

# ---------------------------
# Data Generation Functions
# ---------------------------

def generate_venue_metadata(matches_df: pd.DataFrame) -> Dict:
    """Calculate comprehensive venue metadata including location, characteristics, and historical data."""
    venues = matches_df['venue'].unique()
    venue_metadata = {}
    
    for venue in venues:
        venue_matches = matches_df[matches_df['venue'] == venue]
        
        # Calculate home teams (teams that played most matches at this venue)
        team_matches = pd.concat([
            venue_matches['team1'],
            venue_matches['team2']
        ]).value_counts()
        
        home_teams = team_matches[
            team_matches >= team_matches.max() * 0.4
        ].index.tolist()
        
        # Calculate venue characteristics
        total_matches = len(venue_matches)
        avg_first_innings = venue_matches[venue_matches['win_by_runs'] > 0]['win_by_runs'].mean()
        # Handle NaN values
        if pd.isna(avg_first_innings):
            avg_first_innings = 0
            
        avg_second_innings = venue_matches[venue_matches['win_by_wickets'] > 0]['win_by_wickets'].mean()
        # Handle NaN values
        if pd.isna(avg_second_innings):
            avg_second_innings = 0
            
        batting_first_wins = len(venue_matches[venue_matches['win_by_runs'] > 0])
        chasing_wins = len(venue_matches[venue_matches['win_by_wickets'] > 0])
        
        # Calculate pitch characteristics
        is_high_scoring = avg_first_innings > matches_df['win_by_runs'].replace(0, np.nan).mean()
        favors_chasing = chasing_wins > batting_first_wins
        
        # Generate venue description
        description = f"{venue} has hosted {total_matches} IPL matches. "
        
        if is_high_scoring:
            description += "Known for high-scoring matches, this venue typically produces batting-friendly conditions. "
        else:
            description += "The venue tends to produce moderate scoring matches, offering a good balance between bat and ball. "
        
        if favors_chasing:
            description += "Historical data suggests the venue favors teams chasing, possibly due to dew factor and better batting conditions in the second innings."
        else:
            description += "Teams batting first have had good success here, indicating the pitch might slow down as the match progresses."
        
        # Store metadata
        city = venue_matches['city'].iloc[0] if 'city' in venue_matches.columns else 'Unknown'
        
        venue_metadata[venue] = {
            'total_matches': total_matches,
            'home_teams': home_teams,
            'description': description,
            'city': city,
            'first_match': venue_matches['date'].min(),
            'last_match': venue_matches['date'].max(),
            'characteristics': {
                'is_high_scoring': bool(is_high_scoring),
                'favors_chasing': bool(favors_chasing),
                'avg_first_innings_score': avg_first_innings,
                'avg_second_innings_wickets': avg_second_innings,
                'batting_first_wins': batting_first_wins,
                'chasing_wins': chasing_wins
            }
        }
    
    return venue_metadata

def generate_scoring_patterns(matches_df: pd.DataFrame, deliveries_df: pd.DataFrame, venue: str) -> Dict:
    """Calculate scoring patterns for a specific venue."""
    
    # Get matches at this venue
    venue_matches = matches_df[matches_df['venue'] == venue]
    venue_deliveries = deliveries_df[deliveries_df['match_id'].isin(venue_matches['match_id'])].copy()
    
    if len(venue_deliveries) == 0:
        return {}
    
    # Calculate phase-wise statistics
    def get_phase(over):
        if over <= 6:
            return 'Powerplay'
        elif over <= 15:
            return 'Middle'
        else:
            return 'Death'
    
    # Add over and phase columns - Fixed the DataFrame copy warnings
    venue_deliveries.loc[:, 'over'] = (venue_deliveries['ball'].astype(int) - 1) // 6 + 1
    venue_deliveries.loc[:, 'phase'] = venue_deliveries['over'].apply(get_phase)
    
    # Calculate phase-wise metrics
    phase_metrics = venue_deliveries.groupby(['phase', 'inning']).agg({
        'total_runs': ['sum', 'mean'],
        'is_wicket': 'sum',
        'match_id': 'nunique'
    })
    
    # Calculate over-by-over metrics
    over_metrics = venue_deliveries.groupby(['over', 'inning']).agg({
        'total_runs': ['sum', 'mean'],
        'is_wicket': 'sum',
    })
    
    # Format the data for visualization
    over_data = []
    for idx, data in over_metrics.reset_index().iterrows():
        over_data.append({
            'over': data[('over', '')],
            'inning': f"Inning {data[('inning', '')]}",
            'runs': data[('total_runs', 'mean')],
            'wickets': data[('is_wicket', 'sum')] / data[('total_runs', 'sum')] * 100 if data[('total_runs', 'sum')] > 0 else 0
        })
    
    phase_data = []
    for idx, data in phase_metrics.reset_index().iterrows():
        phase_data.append({
            'phase': data[('phase', '')],
            'inning': f"Inning {data[('inning', '')]}",
            'runs_per_over': data[('total_runs', 'sum')] / (data[('match_id', 'nunique')] * (6 if data[('phase', '')] == 'Powerplay' else 9 if data[('phase', '')] == 'Middle' else 5)),
            'wickets_per_match': data[('is_wicket', 'sum')] / data[('match_id', 'nunique')]
        })
    
    return {
        'over_data': over_data,
        'phase_data': phase_data
    }

def generate_toss_impact(matches_df: pd.DataFrame, venue: str) -> Dict:
    """Calculate toss impact data for a specific venue."""
    venue_matches = matches_df[matches_df['venue'] == venue]
    
    if len(venue_matches) == 0:
        return {}
    
    # Calculate toss decision counts
    toss_decisions = venue_matches['toss_decision'].value_counts().to_dict()
    
    # Calculate win % after winning toss
    toss_winners = venue_matches[venue_matches['toss_winner'] == venue_matches['winner']]
    toss_win_percentage = (len(toss_winners) / len(venue_matches)) * 100
    
    # Calculate bat first vs field first success rates
    bat_first_decisions = venue_matches[venue_matches['toss_decision'] == 'bat']
    field_first_decisions = venue_matches[venue_matches['toss_decision'] == 'field']
    
    bat_first_success = len(bat_first_decisions[bat_first_decisions['toss_winner'] == bat_first_decisions['winner']])
    bat_first_success_rate = (bat_first_success / len(bat_first_decisions)) * 100 if len(bat_first_decisions) > 0 else 0
    
    field_first_success = len(field_first_decisions[field_first_decisions['toss_winner'] == field_first_decisions['winner']])
    field_first_success_rate = (field_first_success / len(field_first_decisions)) * 100 if len(field_first_decisions) > 0 else 0
    
    return {
        'toss_decisions': toss_decisions,
        'toss_win_percentage': toss_win_percentage,
        'bat_first_success_rate': bat_first_success_rate,
        'field_first_success_rate': field_first_success_rate
    }

# ---------------------------
# Display Functions
# ---------------------------

def display_venue_overview(matches_df: pd.DataFrame) -> None:
    """Display overview of all venues with key stats."""
    st.subheader("Venue Overview")
    st.write("Explore the statistics and characteristics of IPL venues across seasons.")
    
    venue_data = generate_venue_metadata(matches_df)
    
    # Create venue selector
    venues = matches_df['venue'].unique()
    selected_venue = st.selectbox("Select a venue", venues, key="venue_overview_selector")
    
    if selected_venue and selected_venue in venue_data:
        venue_info = venue_data[selected_venue]
        
        # Layout the venue information in columns
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown(f"#### {selected_venue}")
            st.markdown(f"**City:** {venue_info['city']}")
            st.markdown(f"**Total Matches:** {venue_info['total_matches']}")
            st.markdown(f"**First Match:** {venue_info['first_match']}")
            st.markdown(f"**Latest Match:** {venue_info['last_match']}")
            
            # Display home teams
            if venue_info['home_teams']:
                st.markdown(f"**Home Teams:** {', '.join(venue_info['home_teams'])}")
            
        with col2:
            # Create a radar chart for venue characteristics
            characteristics = venue_info['characteristics']
            
            # Calculate normalized values between 0 and 1 for the radar chart
            total_matches = matches_df['venue'].value_counts().max()
            batting_wins_ratio = characteristics['batting_first_wins'] / (characteristics['batting_first_wins'] + characteristics['chasing_wins']) if (characteristics['batting_first_wins'] + characteristics['chasing_wins']) > 0 else 0.5
            
            categories = ['Matches Hosted', 'Bat 1st Win %', 'Chase Win %', 'Avg Score', 'High Scoring']
            values = [
                venue_info['total_matches'] / total_matches,
                batting_wins_ratio,
                1 - batting_wins_ratio,
                characteristics['avg_first_innings_score'] / 200,  # Normalize by 200 (high score)
                1 if characteristics['is_high_scoring'] else 0.3
            ]
            
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name=selected_venue,
                line=dict(color="#00AC69") # Using consistent green color
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )
                ),
                showlegend=False,
                margin=dict(l=10, r=10, t=30, b=10),
                height=300,
                template=TEMPLATE,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Display venue description
        st.markdown("#### Venue Characteristics")
        st.write(venue_info['description'])
        
        # Display match history
        st.markdown("#### Recent Matches")
        venue_matches = matches_df[matches_df['venue'] == selected_venue].sort_values('date', ascending=False).head(5)
        
        if not venue_matches.empty:
            for i, match in venue_matches.iterrows():
                winner = match['winner']
                result = f"{match['win_by_runs']} runs" if match['win_by_runs'] > 0 else f"{match['win_by_wickets']} wickets"
                
                st.markdown(f"**{match['date']}**: {match['team1']} vs {match['team2']} - **{winner}** won by {result}")
        else:
            st.info("No matches found for this venue.")
    else:
        st.warning("Please select a venue from the dropdown.")

def display_team_performance(matches_df: pd.DataFrame, deliveries_df: pd.DataFrame) -> None:
    """Display team performance analysis at venues."""
    st.subheader("Team Performance at Venues")
    st.write("Analyze how different teams have performed at various venues over IPL history.")
    
    # Get venue team stats if available
    venue_team_stats_df = load_venue_team_stats()
    
    # Get list of venues and teams
    venues = matches_df['venue'].unique()
    all_teams = pd.concat([matches_df['team1'], matches_df['team2']]).unique()
    
    # Create selectors
    col1, col2 = st.columns(2)
    with col1:
        selected_venue = st.selectbox("Select Venue", venues, key="team_perf_venue")
    with col2:
        selected_team = st.selectbox("Select Team", sorted(all_teams), key="team_perf_team")
    
    # Filter venue matches
    venue_matches = matches_df[matches_df['venue'] == selected_venue]
    
    # Get team's matches at this venue
    team_venue_matches = venue_matches[
        (venue_matches['team1'] == selected_team) | 
        (venue_matches['team2'] == selected_team)
    ]
    
    if len(team_venue_matches) == 0:
        st.info(f"{selected_team} has not played any matches at {selected_venue}.")
        return
    
    # Calculate basic stats
    total_matches = len(team_venue_matches)
    wins = len(team_venue_matches[team_venue_matches['winner'] == selected_team])
    losses = total_matches - wins
    win_percentage = (wins / total_matches) * 100 if total_matches > 0 else 0
    
    # Create win-loss display
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Matches", total_matches)
    with col2:
        st.metric("Wins", wins)
    with col3:
        st.metric("Win Rate", f"{win_percentage:.1f}%")
    
    # Get team deliveries for this venue
    team_venue_deliveries = deliveries_df[
        deliveries_df['match_id'].isin(team_venue_matches['match_id'])
    ]
    
    # Create tabs for different analyses
    batting_tab, bowling_tab, opposition_tab = st.tabs(["Batting", "Bowling", "Opposition Performance"])
    
    # Batting Analysis
    with batting_tab:
        st.markdown(f"#### {selected_team}'s Batting at {selected_venue}")
        
        # Get team's batting deliveries
        batting_deliveries = team_venue_deliveries[
            team_venue_deliveries['batting_team'] == selected_team
        ].copy() # Create a copy to avoid the warning
        
        if len(batting_deliveries) > 0:
            # Calculate batting stats
            total_runs = batting_deliveries['total_runs'].sum()
            matches_batted = batting_deliveries['match_id'].nunique()
            avg_runs_per_match = total_runs / matches_batted if matches_batted > 0 else 0
            
            boundaries = len(batting_deliveries[batting_deliveries['batsman_runs'] >= 4])
            sixes = len(batting_deliveries[batting_deliveries['batsman_runs'] == 6])
            
            # Display metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Avg. Score", f"{avg_runs_per_match:.1f}")
            with col2:
                st.metric("Boundaries", boundaries)
            with col3:
                st.metric("Sixes", sixes)
            
            # Create innings progression chart
            st.markdown("##### Batting Progression")
            
            # Calculate over-by-over stats - Fixed the DataFrame copy warning
            batting_deliveries.loc[:, 'over'] = (batting_deliveries['ball'].astype(int) - 1) // 6 + 1
            over_stats = batting_deliveries.groupby('over').agg({
                'total_runs': 'sum',
                'is_wicket': 'sum',
                'match_id': 'nunique'
            }).reset_index()
            
            over_stats['runs_per_match'] = over_stats['total_runs'] / over_stats['match_id']
            
            fig = px.line(
                over_stats, 
                x='over', 
                y='runs_per_match',
                title=f"Average Runs per Over at {selected_venue}",
                labels={'over': 'Over', 'runs_per_match': 'Avg. Runs'},
                markers=True,
                color_discrete_sequence=["#00AC69"], # Using consistent green color
                template=TEMPLATE
            )
            fig.update_layout(
                xaxis=dict(tickmode='linear', tick0=1, dtick=1),
                margin=dict(l=10, r=10, t=40, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)"
            )
            
            responsive_plotly_chart(fig)
        else:
            st.info(f"No batting data available for {selected_team} at {selected_venue}.")
    
    # Bowling Analysis
    with bowling_tab:
        st.markdown(f"#### {selected_team}'s Bowling at {selected_venue}")
        
        # Get team's bowling deliveries
        bowling_deliveries = team_venue_deliveries[
            team_venue_deliveries['bowling_team'] == selected_team
        ].copy() # Create a copy to avoid the warning
        
        if len(bowling_deliveries) > 0:
            # Calculate bowling stats
            total_wickets = bowling_deliveries['is_wicket'].sum()
            matches_bowled = bowling_deliveries['match_id'].nunique()
            runs_conceded = bowling_deliveries['total_runs'].sum()
            
            avg_wickets_per_match = total_wickets / matches_bowled if matches_bowled > 0 else 0
            economy_rate = (runs_conceded / (len(bowling_deliveries) / 6)) if len(bowling_deliveries) > 0 else 0
            
            # Display metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Avg. Wickets", f"{avg_wickets_per_match:.1f}")
            with col2:
                st.metric("Economy Rate", f"{economy_rate:.2f}")
            with col3:
                st.metric("Total Wickets", total_wickets)
            
            # Create wickets by over chart
            st.markdown("##### Bowling Performance")
            
            # Calculate over-by-over stats - Fixed the DataFrame copy warning
            bowling_deliveries.loc[:, 'over'] = (bowling_deliveries['ball'].astype(int) - 1) // 6 + 1
            over_stats = bowling_deliveries.groupby('over').agg({
                'total_runs': 'sum',
                'is_wicket': 'sum',
                'match_id': 'nunique'
            }).reset_index()
            
            over_stats['wickets_per_match'] = over_stats['is_wicket'] / over_stats['match_id']
            over_stats['runs_per_match'] = over_stats['total_runs'] / over_stats['match_id']
            
            fig = px.bar(
                over_stats, 
                x='over', 
                y='wickets_per_match',
                title=f"Average Wickets per Over at {selected_venue}",
                labels={'over': 'Over', 'wickets_per_match': 'Avg. Wickets'},
                color='runs_per_match',
                color_continuous_scale='RdYlGn_r',
                template=TEMPLATE
            )
            fig.update_layout(
                xaxis=dict(tickmode='linear', tick0=1, dtick=1),
                margin=dict(l=10, r=10, t=40, b=10),
                coloraxis_colorbar=dict(title="Avg. Runs"),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)"
            )
            
            responsive_plotly_chart(fig)
        else:
            st.info(f"No bowling data available for {selected_team} at {selected_venue}.")
    
    # Opposition Analysis
    with opposition_tab:
        st.markdown(f"#### {selected_team}'s Performance vs. Opposition at {selected_venue}")
        
        # Get list of opposition teams
        opposition_teams = []
        for _, match in team_venue_matches.iterrows():
            if match['team1'] == selected_team:
                opposition_teams.append(match['team2'])
            else:
                opposition_teams.append(match['team1'])
        
        # Count matches and wins against each opposition
        opposition_stats = []
        for team in set(opposition_teams):
            matches_vs_team = team_venue_matches[
                ((team_venue_matches['team1'] == selected_team) & (team_venue_matches['team2'] == team)) |
                ((team_venue_matches['team1'] == team) & (team_venue_matches['team2'] == selected_team))
            ]
            total = len(matches_vs_team)
            wins = len(matches_vs_team[matches_vs_team['winner'] == selected_team])
            
            opposition_stats.append({
                'team': team,
                'matches': total,
                'wins': wins,
                'losses': total - wins,
                'win_rate': (wins / total) * 100 if total > 0 else 0
            })
        
        if opposition_stats:
            # Convert to DataFrame and sort
            opposition_df = pd.DataFrame(opposition_stats)
            opposition_df = opposition_df.sort_values('matches', ascending=False)
            
            # Create horizontal bar chart
            fig = px.bar(
                opposition_df,
                y='team',
                x='matches',
                color='win_rate',
                labels={'team': 'Opposition', 'matches': 'Matches Played', 'win_rate': 'Win Rate (%)'},
                color_continuous_scale='RdYlGn',
                range_color=[0, 100],
                text='wins',
                orientation='h',
                template=TEMPLATE
            )
            
            fig.update_traces(texttemplate='%{text} wins', textposition='inside')
            fig.update_layout(
                title=f"{selected_team}'s Record vs. Opposition at {selected_venue}",
                margin=dict(l=10, r=10, t=40, b=10),
                height=400,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)"
            )
            
            responsive_plotly_chart(fig)
        else:
            st.info(f"No opposition data available for {selected_team} at {selected_venue}.")

def display_scoring_patterns(matches_df: pd.DataFrame, deliveries_df: pd.DataFrame) -> None:
    """Display scoring patterns at venues."""
    st.subheader("Venue Scoring Patterns")
    st.write("Analyze how runs are scored and wickets fall across different phases of the game at each venue.")
    
    # Get list of venues
    venues = matches_df['venue'].unique()
    selected_venue = st.selectbox("Select Venue", venues, key="scoring_patterns_venue")
    
    # Generate scoring patterns for selected venue
    scoring_patterns = generate_scoring_patterns(matches_df, deliveries_df, selected_venue)
    
    if not scoring_patterns:
        st.info(f"No detailed scoring data available for {selected_venue}.")
        return
    
    # Create tabs for different analyses
    over_tab, phase_tab = st.tabs(["Over-by-Over Analysis", "Phase Analysis"])
    
    # Over-by-Over Analysis
    with over_tab:
        st.markdown(f"#### Over-by-Over Scoring at {selected_venue}")
        
        # Convert data to DataFrame
        over_df = pd.DataFrame(scoring_patterns['over_data'])
        
        # Create line chart for runs by over
        fig = px.line(
            over_df, 
            x='over', 
            y='runs', 
            color='inning',
            title=f"Average Runs per Over at {selected_venue}",
            labels={'over': 'Over', 'runs': 'Avg. Runs', 'inning': 'Inning'},
            markers=True,
            line_shape='spline',
            color_discrete_sequence=COLOR_SEQUENCE,
            template=TEMPLATE
        )
        
        fig.update_layout(
            xaxis=dict(tickmode='linear', tick0=1, dtick=1),
            margin=dict(l=10, r=10, t=40, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        
        responsive_plotly_chart(fig)
        
        # Create bar chart for wickets by over
        wicket_fig = px.bar(
            over_df, 
            x='over', 
            y='wickets', 
            color='inning',
            barmode='group',
            title=f"Wicket Likelihood by Over at {selected_venue} (%)",
            labels={'over': 'Over', 'wickets': 'Wicket Probability (%)', 'inning': 'Inning'},
            color_discrete_sequence=COLOR_SEQUENCE,
            template=TEMPLATE
        )
        
        wicket_fig.update_layout(
            xaxis=dict(tickmode='linear', tick0=1, dtick=1),
            margin=dict(l=10, r=10, t=40, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        
        responsive_plotly_chart(wicket_fig)
    
    # Phase Analysis
    with phase_tab:
        st.markdown(f"#### Phase-wise Analysis at {selected_venue}")
        
        # Convert data to DataFrame
        phase_df = pd.DataFrame(scoring_patterns['phase_data'])
        
        # Create column layout
        col1, col2 = st.columns(2)
        
        with col1:
            # Create bar chart for runs by phase
            phase_fig = px.bar(
                phase_df, 
                x='phase', 
                y='runs_per_over', 
                color='inning',
                barmode='group',
                title=f"Runs per Over by Phase at {selected_venue}",
                labels={'phase': 'Phase', 'runs_per_over': 'Runs per Over', 'inning': 'Inning'},
                category_orders={"phase": ["Powerplay", "Middle", "Death"]},
                color_discrete_sequence=COLOR_SEQUENCE,
                template=TEMPLATE
            )
            
            phase_fig.update_layout(
                margin=dict(l=10, r=10, t=40, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)"
            )
            
            st.plotly_chart(phase_fig, use_container_width=True)
        
        with col2:
            # Create bar chart for wickets by phase
            wicket_phase_fig = px.bar(
                phase_df, 
                x='phase', 
                y='wickets_per_match', 
                color='inning',
                barmode='group',
                title=f"Wickets per Match by Phase at {selected_venue}",
                labels={'phase': 'Phase', 'wickets_per_match': 'Wickets per Match', 'inning': 'Inning'},
                category_orders={"phase": ["Powerplay", "Middle", "Death"]},
                color_discrete_sequence=COLOR_SEQUENCE,
                template=TEMPLATE
            )
            
            wicket_phase_fig.update_layout(
                margin=dict(l=10, r=10, t=40, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)"
            )
            
            st.plotly_chart(wicket_phase_fig, use_container_width=True)
        
        # Add phase description
        st.markdown("""
        **Phase Definitions:**
        - **Powerplay**: Overs 1-6
        - **Middle**: Overs 7-15
        - **Death**: Overs 16-20
        """)

def display_toss_analysis(matches_df: pd.DataFrame) -> None:
    """Display toss impact analysis for venues."""
    st.subheader("Toss Impact Analysis")
    st.write("Analyze how winning the toss affects match outcomes at different venues.")
    
    # Get list of venues
    venues = matches_df['venue'].unique()
    selected_venue = st.selectbox("Select Venue", venues, key="toss_analysis_venue")
    
    # Generate toss impact data for selected venue
    toss_data = generate_toss_impact(matches_df, selected_venue)
    
    if not toss_data:
        st.info(f"No toss data available for {selected_venue}.")
        return
    
    # Create layout
    col1, col2 = st.columns(2)
    
    with col1:
        # Display toss win percentage
        st.metric(
            "Matches Won After Winning Toss",
            f"{toss_data['toss_win_percentage']:.1f}%"
        )
        
        # Create pie chart for toss decisions
        decisions = toss_data['toss_decisions']
        
        decision_labels = []
        decision_values = []
        
        for decision, count in decisions.items():
            decision_labels.append(decision.capitalize())
            decision_values.append(count)
        
        decision_fig = go.Figure(data=[go.Pie(
            labels=decision_labels,
            values=decision_values,
            hole=.4,
            marker_colors=['#00AC69', '#4361EE'] # Using consistent app colors
        )])
        
        decision_fig.update_layout(
            title_text="Toss Decisions",
            margin=dict(l=10, r=10, t=40, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        
        st.plotly_chart(decision_fig, use_container_width=True)
    
    with col2:
        # Create success rate comparison
        success_data = {
            'Decision': ['Bat First', 'Field First'],
            'Success Rate': [
                toss_data['bat_first_success_rate'],
                toss_data['field_first_success_rate']
            ]
        }
        
        success_df = pd.DataFrame(success_data)
        
        success_fig = px.bar(
            success_df,
            x='Decision',
            y='Success Rate',
            color='Decision',
            title="Success Rate by Toss Decision",
            labels={'Success Rate': 'Win Percentage (%)'},
            text_auto='.1f',
            color_discrete_sequence=['#00AC69', '#4361EE'], # Using consistent app colors
            template=TEMPLATE
        )
        
        success_fig.update_traces(texttemplate='%{text}%', textposition='outside')
        success_fig.update_layout(
            margin=dict(l=10, r=10, t=40, b=10),
            showlegend=False,
            yaxis=dict(range=[0, 100]),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        
        st.plotly_chart(success_fig, use_container_width=True)
    
    # Display insights
    st.markdown("#### Toss Impact Insights")
    
    # Generate insights based on the data
    if toss_data['toss_win_percentage'] > 60:
        st.markdown(f"💡 **Winning the toss is highly advantageous at {selected_venue}**, with teams winning {toss_data['toss_win_percentage']:.1f}% of matches after winning the toss.")
    elif toss_data['toss_win_percentage'] < 40:
        st.markdown(f"💡 **Interestingly, winning the toss seems to be a disadvantage at {selected_venue}**, with teams losing {100-toss_data['toss_win_percentage']:.1f}% of matches after winning the toss.")
    else:
        st.markdown(f"💡 **The toss has a moderate impact at {selected_venue}**, with toss winners winning {toss_data['toss_win_percentage']:.1f}% of matches.")
    
    # Decision insights
    bat_field_diff = abs(toss_data['bat_first_success_rate'] - toss_data['field_first_success_rate'])
    if bat_field_diff > 20:
        better_choice = "batting first" if toss_data['bat_first_success_rate'] > toss_data['field_first_success_rate'] else "fielding first"
        st.markdown(f"💡 **The data strongly suggests that {better_choice} is the better choice after winning the toss** at {selected_venue}.")
    else:
        st.markdown(f"💡 **There is no clear advantage to either batting or fielding first** at {selected_venue} based on historical data.")

def display_venue_analysis(matches_df: pd.DataFrame, deliveries_df: pd.DataFrame) -> None:
    """Display comprehensive venue analysis with all enhanced features."""
    st.title("📍 Venue Analysis")
    st.write("Explore in-depth statistics and patterns across different venues in the IPL.")
    
    # Create tabs for different analyses
    tabs = st.tabs([
        "Venue Overview",
        "Team Performance",
        "Scoring Patterns",
        "Toss Impact"
    ])
    
    # Venue Overview Tab
    with tabs[0]:
        display_venue_overview(matches_df)
    
    # Team Performance Tab
    with tabs[1]:
        display_team_performance(matches_df, deliveries_df)
    
    # Scoring Patterns Tab
    with tabs[2]:
        display_scoring_patterns(matches_df, deliveries_df)
    
    # Toss Impact Tab
    with tabs[3]:
        display_toss_analysis(matches_df) 