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
        dict: Dictionary containing the loaded data.
    """
    data_dir = Path(__file__).resolve().parent.parent / "data"
    
    # Convert filename from .json to .parquet if needed
    if filename.endswith('.json'):
        parquet_filename = filename.replace('.json', '.parquet')
    else:
        parquet_filename = filename
        
    file_path = data_dir / parquet_filename
    
    if not os.path.exists(file_path):
        # Try to fall back to JSON if Parquet doesn't exist
        json_path = data_dir / filename.replace('.parquet', '.json')
        if os.path.exists(json_path):
            return load_precomputed_json(filename.replace('.parquet', '.json'))
        st.error(f"Precomputed data file not found: {parquet_filename}")
        return {}
    
    try:
        df = pd.read_parquet(file_path)
        # Convert DataFrame to dict
        if len(df) == 1:
            # If it's a single row DataFrame (converted from a dict)
            return df.iloc[0].to_dict()
        else:
            # If it's a multi-row DataFrame (converted from a list)
            return df.to_dict(orient='records')
    except Exception as e:
        st.error(f"Error loading {parquet_filename}: {str(e)}")
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

# ---------------------------
# Legacy Calculation Functions (Kept for reference)
# ---------------------------

def calculate_venue_metadata(matches_df: pd.DataFrame) -> Dict:
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
        avg_second_innings = venue_matches[venue_matches['win_by_wickets'] > 0]['win_by_wickets'].mean()
        batting_first_wins = len(venue_matches[venue_matches['win_by_runs'] > 0])
        chasing_wins = len(venue_matches[venue_matches['win_by_wickets'] > 0])
        
        # Calculate pitch characteristics
        is_high_scoring = avg_first_innings > matches_df['win_by_runs'].mean()
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
        venue_metadata[venue] = {
            'total_matches': total_matches,
            'home_teams': home_teams,
            'description': description,
            'city': venue_matches['city'].iloc[0] if 'city' in venue_matches.columns else 'Unknown',
            'state': venue_matches['state'].iloc[0] if 'state' in venue_matches.columns else 'Unknown',
            'first_match': venue_matches['date'].min(),
            'last_match': venue_matches['date'].max(),
            'characteristics': {
                'is_high_scoring': is_high_scoring,
                'favors_chasing': favors_chasing,
                'avg_first_innings_score': avg_first_innings,
                'avg_second_innings_wickets': avg_second_innings,
                'batting_first_wins': batting_first_wins,
                'chasing_wins': chasing_wins
            }
        }
    
    return venue_metadata

def calculate_team_venue_stats(matches_df: pd.DataFrame, deliveries_df: pd.DataFrame) -> Dict:
    """Calculate comprehensive team performance statistics at each venue."""
    venues = matches_df['venue'].unique()
    teams = pd.concat([matches_df['team1'], matches_df['team2']]).unique()
    venue_stats = {}
    
    for venue in venues:
        venue_matches = matches_df[matches_df['venue'] == venue]
        venue_stats[venue] = {}
        
        for team in teams:
            # Get matches where team played at this venue
            team_matches = venue_matches[
                (venue_matches['team1'] == team) |
                (venue_matches['team2'] == team)
            ]
            
            if len(team_matches) == 0:
                continue
            
            # Calculate W/L/T record
            wins = len(team_matches[team_matches['winner'] == team])
            total = len(team_matches)
            losses = total - wins
            
            # Calculate batting stats by innings
            team_deliveries = deliveries_df[
                deliveries_df['match_id'].isin(team_matches['match_id'])
            ]
            
            batting_stats = team_deliveries[
                team_deliveries['batting_team'] == team
            ].groupby('inning').agg({
                'total_runs': ['mean', 'std', 'sum'],
                'is_wicket': 'sum',
                'match_id': 'nunique'
            })
            
            bowling_stats = team_deliveries[
                team_deliveries['bowling_team'] == team
            ].groupby('inning').agg({
                'total_runs': ['mean', 'std', 'sum'],
                'is_wicket': 'sum',
                'match_id': 'nunique'
            })
            
            # Calculate home advantage
            is_home_team = team in venue_stats[venue].get('home_teams', [])
            home_matches = team_matches[
                (team_matches['team1'] == team) & (venue_matches['venue'] == venue)
            ]
            home_wins = len(home_matches[home_matches['winner'] == team])
            home_win_pct = (home_wins / len(home_matches)) * 100 if len(home_matches) > 0 else 0
            
            venue_stats[venue][team] = {
                'matches': total,
                'wins': wins,
                'losses': losses,
                'win_percentage': (wins / total) * 100,
                'batting_stats': batting_stats.to_dict(),
                'bowling_stats': bowling_stats.to_dict(),
                'is_home_team': is_home_team,
                'home_win_percentage': home_win_pct
            }
    
    return venue_stats

def calculate_scoring_patterns(matches_df: pd.DataFrame, deliveries_df: pd.DataFrame) -> Dict:
    """Calculate scoring patterns for a venue."""
    
    # Calculate phase-wise statistics
    def get_phase(over):
        if over <= 6:
            return 'Powerplay'
        elif over <= 15:
            return 'Middle'
        else:
            return 'Death'
    
    # Add phase and innings information
    deliveries_df['phase'] = deliveries_df['over'].apply(get_phase)
    
    # Calculate phase-wise statistics
    phase_stats = deliveries_df.groupby(['inning', 'phase']).agg({
        'total_runs': ['mean', 'std', 'sum', 'count']
    })
    
    # Calculate innings-wise statistics
    innings_stats = deliveries_df.groupby('inning').agg({
        'total_runs': ['mean', 'std', 'sum', 'count']
    })
    
    return {
        'phase_stats': phase_stats,
        'innings_stats': innings_stats
    }

def calculate_toss_impact(matches_df: pd.DataFrame) -> Dict:
    """Calculate comprehensive toss impact statistics for each venue."""
    venues = matches_df['venue'].unique()
    toss_stats = {}
    
    for venue in venues:
        venue_matches = matches_df[matches_df['venue'] == venue]
        
        # Basic toss decisions
        toss_decisions = venue_matches['toss_decision'].value_counts()
        toss_decisions_pct = (toss_decisions / len(venue_matches)) * 100
        
        # Toss win impact
        toss_wins = venue_matches['toss_winner'] == venue_matches['winner']
        toss_win_pct = (toss_wins.sum() / len(venue_matches)) * 100
        
        # Team-wise toss preferences and success
        team_toss_stats = {}
        for team in pd.concat([venue_matches['team1'], venue_matches['team2']]).unique():
            team_tosses = venue_matches[venue_matches['toss_winner'] == team]
            if not team_tosses.empty:
                team_toss_stats[team] = {
                    'decisions': team_tosses['toss_decision'].value_counts().to_dict(),
                    'success_rate': (
                        len(team_tosses[team_tosses['winner'] == team]) / 
                        len(team_tosses)
                    ) * 100
                }
        
        # Toss decision success by innings
        bat_first_success = len(
            venue_matches[
                (venue_matches['toss_decision'] == 'bat') & 
                (venue_matches['toss_winner'] == venue_matches['winner'])
            ]
        )
        field_first_success = len(
            venue_matches[
                (venue_matches['toss_decision'] == 'field') & 
                (venue_matches['toss_winner'] == venue_matches['winner'])
            ]
        )
        
        toss_stats[venue] = {
            'toss_decisions': toss_decisions.to_dict(),
            'toss_decisions_pct': toss_decisions_pct.to_dict(),
            'toss_win_pct': toss_win_pct,
            'team_stats': team_toss_stats,
            'decision_success': {
                'bat_first': bat_first_success,
                'field_first': field_first_success,
                'bat_first_pct': (bat_first_success / len(venue_matches)) * 100,
                'field_first_pct': (field_first_success / len(venue_matches)) * 100
            }
        }
    
    return toss_stats

def calculate_weather_impact(matches_df: pd.DataFrame) -> Dict:
    """Calculate weather impact statistics for venues."""
    venues = matches_df['venue'].unique()
    weather_stats = {}
    
    for venue in venues:
        venue_matches = matches_df[matches_df['venue'] == venue]
        total_matches = len(venue_matches)
        
        # Calculate rain-affected matches (if result column exists)
        if 'result' in venue_matches.columns:
            rain_affected = len(venue_matches[venue_matches['result'].str.contains('rain', case=False, na=False)])
        else:
            rain_affected = 0
        
        # Calculate match timing distribution (if start_time column exists)
        if 'start_time' in venue_matches.columns:
            evening_matches = len(venue_matches[venue_matches['start_time'].str.contains('19:30|20:00', na=False)])
            day_matches = total_matches - evening_matches
        else:
            # If no timing data available, assume equal distribution
            evening_matches = total_matches // 2
            day_matches = total_matches - evening_matches
        
        weather_stats[venue] = {
            'total_matches': total_matches,
            'rain_affected_matches': rain_affected,
            'rain_percentage': (rain_affected / total_matches * 100) if total_matches > 0 else 0,
            'evening_matches': evening_matches,
            'day_matches': day_matches,
            'evening_percentage': (evening_matches / total_matches * 100) if total_matches > 0 else 0
        }
    
    return weather_stats

def calculate_batting_stats(venue_matches: pd.DataFrame, deliveries_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate batting statistics for teams at a venue."""
    # Get deliveries for the venue matches
    venue_deliveries = deliveries_df[
        deliveries_df['match_id'].isin(venue_matches['match_id'])
    ]
    
    # Calculate team batting stats
    batting_stats = venue_deliveries.groupby('batting_team').agg({
        'total_runs': ['sum', 'mean', 'max'],
        'match_id': 'nunique',
        'ball': lambda x: len(x) / 6.0  # Convert balls to overs
    }).round(2)
    
    # Flatten column names
    batting_stats.columns = ['total_runs', 'average_score', 'highest_score', 'innings', 'overs']
    
    # Calculate strike rate
    batting_stats['strike_rate'] = (batting_stats['total_runs'] / batting_stats['overs']).round(2)
    
    return batting_stats

def calculate_bowling_stats(venue_matches: pd.DataFrame, deliveries_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate bowling statistics for teams at a venue."""
    # Get deliveries for the venue matches
    venue_deliveries = deliveries_df[
        deliveries_df['match_id'].isin(venue_matches['match_id'])
    ]
    
    # Calculate team bowling stats
    bowling_stats = venue_deliveries.groupby('bowling_team').agg({
        'is_wicket': ['sum', 'mean'],
        'match_id': 'nunique',
        'ball': lambda x: len(x) / 6.0,  # Convert balls to overs
        'total_runs': 'sum'
    }).round(2)
    
    # Flatten column names
    bowling_stats.columns = ['wickets', 'wickets_per_over', 'innings', 'overs', 'runs_conceded']
    
    # Calculate economy rate
    bowling_stats['economy'] = (bowling_stats['runs_conceded'] / bowling_stats['overs']).round(2)
    
    # Calculate average wickets per match
    bowling_stats['average_wickets'] = (bowling_stats['wickets'] / bowling_stats['innings']).round(2)
    
    # Find best bowling performances
    best_bowling = venue_deliveries[venue_deliveries['is_wicket'] == 1].groupby(
        ['match_id', 'bowling_team']
    ).agg({
        'is_wicket': 'sum',
        'total_runs': 'sum'
    })
    
    best_bowling = best_bowling.reset_index().groupby('bowling_team').apply(
        lambda x: f"{int(x['is_wicket'].max())}/{int(x['total_runs'].min())}"
    ).to_dict()
    
    bowling_stats['best_bowling'] = bowling_stats.index.map(best_bowling)
    
    return bowling_stats

# ---------------------------
# Display Functions
# ---------------------------

def display_venue_overview(matches_df: pd.DataFrame) -> None:
    """Display comprehensive venue overview with metadata and characteristics."""
    st.header("Venue Overview")
    
    # Load precomputed venue metadata
    venue_metadata = load_venue_metadata()
    
    if not venue_metadata:
        st.error("Venue metadata not available. Please run the data preprocessing script.")
        return
    
    # Create venue selector with unique key
    selected_venue = st.selectbox(
        "Select Venue",
        options=sorted(venue_metadata.keys()),
        key="venue_overview_selector"
    )
    
    metadata = venue_metadata[selected_venue]
    
    # Display basic information
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Venue Information")
        st.metric("Total Matches", metadata['total_matches'])
        st.metric("City", metadata['city'])
        st.metric("State", metadata['state'])
        st.metric("First IPL Match", str(metadata['first_match'])[:10])
        st.metric("Latest IPL Match", str(metadata['last_match'])[:10])
        
        st.subheader("Home Teams")
        for team in metadata['home_teams']:
            st.write(f"• {team}")
    
    with col2:
        st.subheader("Venue Characteristics")
        st.write(metadata['description'])
        
        # Display pitch characteristics
        chars = metadata['characteristics']
        try:
            avg_score = float(chars['avg_first_innings_score'])
            st.metric("Average First Innings Score", f"{avg_score:.1f}")
        except (TypeError, ValueError, KeyError):
            st.metric("Average First Innings Score", "N/A")
            
        try:
            bat_first_wins = float(chars['batting_first_wins'])
            total_matches = float(metadata['total_matches'])
            win_pct = (bat_first_wins / total_matches) * 100 if total_matches > 0 else 0
            st.metric("Batting First Win %", f"{win_pct:.1f}%")
        except (TypeError, ValueError, KeyError, ZeroDivisionError):
            st.metric("Batting First Win %", "N/A")
        
        # Pitch type indicator
        if chars['is_high_scoring']:
            st.success("🏏 Batting-friendly pitch")
        else:
            st.warning("🎯 Bowling-friendly pitch")
        
        if chars['favors_chasing']:
            st.info("🎯 Favors chasing teams")
        else:
            st.info("🏏 Favors batting first")

def display_team_performance(matches_df: pd.DataFrame, deliveries_df: pd.DataFrame) -> None:
    """Display team performance analysis for each venue."""
    st.header("Team Performance Analysis")
    
    # Load precomputed team performance at venues
    venue_team_performance = load_venue_team_performance()
    
    if not venue_team_performance:
        st.error("Team performance data not available. Please run the data preprocessing script.")
        return
    
    # Get selected venue with unique key
    venue = st.selectbox(
        "Select Venue", 
        sorted(venue_team_performance.keys()),
        key="team_performance_venue_selector"
    )
    
    if venue not in venue_team_performance:
        st.warning(f"No team performance data available for {venue}")
        return
    
    # Extract team performance data for the selected venue
    team_data = venue_team_performance[venue]
    
    # Create DataFrames for batting and bowling stats
    batting_stats = []
    bowling_stats = []
    
    for team, stats in team_data.items():
        # Check if stats is None
        if stats is None:
            # Add default values if stats is None
            batting_stats.append({
                'batting_team': team,
                'total_runs': 0,
                'average_score': 0,
                'highest_score': 0,
                'innings': 0,
                'overs': 0,
                'strike_rate': 0
            })
            
            bowling_stats.append({
                'bowling_team': team,
                'wickets': 0,
                'average_wickets': 0,
                'innings': 0,
                'overs': 0,
                'runs_conceded': 0,
                'economy': 0,
                'best_bowling': '0/0'
            })
            continue
            
        # Check if batting_stats exists and is not None
        if 'batting_stats' in stats and stats['batting_stats'] is not None:
            batting_stats.append({
                'batting_team': team,
                'total_runs': stats['batting_stats'].get('total_runs', 0),
                'average_score': stats['batting_stats'].get('total_runs', 0) / stats.get('matches', 1) if stats.get('matches', 0) > 0 else 0,
                'highest_score': stats['batting_stats'].get('highest_score', 0),
                'innings': stats.get('matches', 0),
                'overs': stats['batting_stats'].get('overs', 0),
                'strike_rate': stats['batting_stats'].get('batting_sr', 0)
            })
        else:
            # Add default values if batting_stats is missing
            batting_stats.append({
                'batting_team': team,
                'total_runs': 0,
                'average_score': 0,
                'highest_score': 0,
                'innings': stats.get('matches', 0),
                'overs': 0,
                'strike_rate': 0
            })
        
        # Check if bowling_stats exists and is not None
        if 'bowling_stats' in stats and stats['bowling_stats'] is not None:
            bowling_stats.append({
                'bowling_team': team,
                'wickets': stats['bowling_stats'].get('wickets', 0),
                'average_wickets': stats['bowling_stats'].get('wickets', 0) / stats.get('matches', 1) if stats.get('matches', 0) > 0 else 0,
                'innings': stats.get('matches', 0),
                'overs': stats['bowling_stats'].get('overs', 0),
                'runs_conceded': stats['bowling_stats'].get('runs_conceded', 0),
                'economy': stats['bowling_stats'].get('economy', 0),
                'best_bowling': stats['bowling_stats'].get('best_bowling', '0/0')
            })
        else:
            # Add default values if bowling_stats is missing
            bowling_stats.append({
                'bowling_team': team,
                'wickets': 0,
                'average_wickets': 0,
                'innings': stats.get('matches', 0),
                'overs': 0,
                'runs_conceded': 0,
                'economy': 0,
                'best_bowling': '0/0'
            })
    
    # Convert to DataFrames
    batting_df = pd.DataFrame(batting_stats)
    bowling_df = pd.DataFrame(bowling_stats)
    
    # Display team win/loss record
    st.subheader(f"Team Records at {venue}")
    
    # Create a DataFrame for team records
    team_records = []
    for team, stats in team_data.items():
        if stats is None:
            team_records.append({
                'Team': team,
                'Matches': 0,
                'Wins': 0,
                'Losses': 0,
                'Win %': 0
            })
        else:
            team_records.append({
                'Team': team,
                'Matches': stats.get('matches', 0),
                'Wins': stats.get('wins', 0),
                'Losses': stats.get('losses', 0),
                'Win %': (stats.get('wins', 0) / stats.get('matches', 1) * 100) if stats.get('matches', 0) > 0 else 0
            })
    
    team_records_df = pd.DataFrame(team_records)
    team_records_df = team_records_df.sort_values('Matches', ascending=False)
    
    # Display team records
    st.dataframe(
        team_records_df.style.format({
            'Win %': '{:.1f}%'
        }),
        use_container_width=True
    )
    
    # Create tabs for batting and bowling analysis
    tabs = st.tabs(["Batting Analysis", "Bowling Analysis"])
    
    # Batting Analysis Tab
    with tabs[0]:
        if not batting_df.empty:
            # Sort by total runs
            batting_df = batting_df.sort_values('total_runs', ascending=False)
            
            # Display batting stats
            st.subheader("Batting Performance")
            st.dataframe(
                batting_df.style.format({
                    'total_runs': '{:.0f}',
                    'average_score': '{:.1f}',
                    'highest_score': '{:.0f}',
                    'overs': '{:.1f}',
                    'strike_rate': '{:.2f}'
                }),
                use_container_width=True
            )
            
            # Create bar chart for total runs
            fig = px.bar(
                batting_df,
                x='batting_team',
                y='total_runs',
                title='Total Runs by Team',
                color='batting_team',
                template='plotly_dark'
            )
            fig.update_layout(
                xaxis_title='Team',
                yaxis_title='Total Runs',
                showlegend=False,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                yaxis=dict(gridcolor='rgba(128,128,128,0.1)'),
                xaxis=dict(gridcolor='rgba(128,128,128,0.1)')
            )
            responsive_plotly_chart(fig, use_container_width=True)
            
            # Create bar chart for average score
            fig = px.bar(
                batting_df,
                x='batting_team',
                y='average_score',
                title='Average Score by Team',
                color='batting_team',
                template='plotly_dark'
            )
            fig.update_layout(
                xaxis_title='Team',
                yaxis_title='Average Score',
                showlegend=False,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                yaxis=dict(gridcolor='rgba(128,128,128,0.1)'),
                xaxis=dict(gridcolor='rgba(128,128,128,0.1)')
            )
            responsive_plotly_chart(fig, use_container_width=True)
        else:
            st.info("No batting data available for this venue")
    
    # Bowling Analysis Tab
    with tabs[1]:
        if not bowling_df.empty:
            # Sort by wickets
            bowling_df = bowling_df.sort_values('wickets', ascending=False)
            
            # Display bowling stats
            st.subheader("Bowling Performance")
            st.dataframe(
                bowling_df.style.format({
                    'wickets': '{:.0f}',
                    'average_wickets': '{:.1f}',
                    'overs': '{:.1f}',
                    'runs_conceded': '{:.0f}',
                    'economy': '{:.2f}'
                }),
                use_container_width=True
            )
            
            # Create bar chart for wickets
            fig = px.bar(
                bowling_df,
                x='bowling_team',
                y='wickets',
                title='Total Wickets by Team',
                color='bowling_team',
                template='plotly_dark'
            )
            fig.update_layout(
                xaxis_title='Team',
                yaxis_title='Wickets',
                showlegend=False,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                yaxis=dict(gridcolor='rgba(128,128,128,0.1)'),
                xaxis=dict(gridcolor='rgba(128,128,128,0.1)')
            )
            responsive_plotly_chart(fig, use_container_width=True)
            
            # Create bar chart for economy
            fig = px.bar(
                bowling_df,
                x='bowling_team',
                y='economy',
                title='Economy Rate by Team',
                color='bowling_team',
                template='plotly_dark'
            )
            fig.update_layout(
                xaxis_title='Team',
                yaxis_title='Economy Rate',
                showlegend=False,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                yaxis=dict(gridcolor='rgba(128,128,128,0.1)'),
                xaxis=dict(gridcolor='rgba(128,128,128,0.1)')
            )
            responsive_plotly_chart(fig, use_container_width=True)
        else:
            st.info("No bowling data available for this venue")

def display_scoring_patterns(matches_df: pd.DataFrame, deliveries_df: pd.DataFrame):
    st.header("Scoring Patterns")
    
    # Load precomputed scoring patterns
    venue_scoring_patterns = load_venue_scoring_patterns()
    
    if not venue_scoring_patterns:
        st.error("Scoring patterns data not available. Please run the data preprocessing script.")
        return
    
    # Get unique venues for selection
    venues = sorted(venue_scoring_patterns.keys())
    selected_venue = st.selectbox("Select Venue", venues, key="scoring_patterns_venue_selector")
    
    if selected_venue not in venue_scoring_patterns:
        st.warning(f"No scoring pattern data available for {selected_venue}")
        return
    
    # Extract scoring patterns for the selected venue
    venue_patterns = venue_scoring_patterns[selected_venue]
    
    # Display phase-wise scoring
    st.subheader("Scoring by Match Phase")
    
    # Convert phase stats to DataFrame
    phase_df = pd.DataFrame(venue_patterns['phase_stats'])
    
    if not phase_df.empty:
        # Create phase-wise bar chart
        fig = px.bar(phase_df, 
                    x='phase', 
                    y='mean',
                    color='inning',
                    barmode='group',
                    title=f"Average Runs by Phase at {selected_venue}",
                    labels={
                        'phase': "Phase",
                        'mean': "Average Runs",
                        'inning': "Innings"
                    })
        responsive_plotly_chart(fig)
    else:
        st.info("No phase-wise scoring data available for this venue")
    
    # Display innings comparison
    st.subheader("Innings Comparison")
    
    # Convert innings stats to DataFrame
    innings_df = pd.DataFrame(venue_patterns['innings_stats'])
    
    if not innings_df.empty:
        # Create box plot for innings comparison
        fig = px.box(innings_df,
                    x='inning',
                    y='sum',
                    title=f"Run Distribution by Innings at {selected_venue}",
                    labels={
                        'inning': "Innings",
                        'sum': "Total Runs"
                    })
        responsive_plotly_chart(fig)
    else:
        st.info("No innings comparison data available for this venue")
    
    # Display score distribution
    st.subheader("Score Distribution")
    
    # Extract match totals
    match_totals = venue_patterns['match_totals']
    
    if match_totals['values']:
        # Create histogram
        fig = px.histogram(x=match_totals['values'],
                        nbins=20,
                        title=f"Match Score Distribution at {selected_venue}",
                        labels={'x': 'Total Match Score', 'y': 'Frequency'})
        responsive_plotly_chart(fig)
        
        # Display summary statistics
        st.write("Summary Statistics:")
        stats = {
            "Mean Score": f"{match_totals['mean']:.2f}",
            "Median Score": f"{match_totals['median']:.2f}",
            "Std Dev": f"{match_totals['std']:.2f}",
            "Minimum": f"{match_totals['min']:.2f}",
            "Maximum": f"{match_totals['max']:.2f}"
        }
        st.json(stats)
    else:
        st.info("No match score distribution data available for this venue")

def display_toss_analysis(matches_df: pd.DataFrame) -> None:
    """Display comprehensive toss impact analysis for venues."""
    st.header("Toss Impact Analysis")
    
    # Load precomputed toss impact
    venue_toss_impact = load_venue_toss_impact()
    
    if not venue_toss_impact:
        st.error("Toss impact data not available. Please run the data preprocessing script.")
        return
    
    # Create venue selector with unique key
    selected_venue = st.selectbox(
        "Select Venue",
        options=sorted(venue_toss_impact.keys()),
        key="toss_analysis_venue_selector"
    )
    
    if selected_venue not in venue_toss_impact:
        st.warning(f"No toss impact data available for {selected_venue}")
        return
    
    stats = venue_toss_impact[selected_venue]
    
    # Create tabs for different analyses
    tabs = st.tabs([
        "Toss Decisions",
        "Team Preferences",
        "Success Analysis"
    ])
    
    # Toss Decisions Tab
    with tabs[0]:
        col1, col2 = st.columns(2)
        
        with col1:
            # Toss decisions pie chart
            if stats['toss_decisions']:
                fig = px.pie(
                    values=list(stats['toss_decisions'].values()),
                    names=list(stats['toss_decisions'].keys()),
                    title="Toss Decisions Distribution"
                )
                responsive_plotly_chart(fig, use_container_width=True)
            else:
                st.info("No toss decision data available")
        
        with col2:
            try:
                toss_win_pct = float(stats['toss_win_pct'])
                st.metric("Toss Win Impact", f"{toss_win_pct:.1f}%")
            except (TypeError, ValueError, KeyError):
                st.metric("Toss Win Impact", "N/A")
                
            try:
                bat_first_pct = float(stats['decision_success']['bat_first_pct'])
                st.metric("Bat First Success", f"{bat_first_pct:.1f}%")
            except (TypeError, ValueError, KeyError):
                st.metric("Bat First Success", "N/A")
    
    # Team Preferences Tab
    with tabs[1]:
        st.subheader("Team-wise Toss Preferences")
        
        if stats['team_stats']:
            for team, team_stats in stats['team_stats'].items():
                st.write(f"**{team}**")
                col1, col2 = st.columns(2)
                
                with col1:
                    # Team's toss decisions
                    if team_stats['decisions']:
                        fig = px.pie(
                            values=list(team_stats['decisions'].values()),
                            names=list(team_stats['decisions'].keys()),
                            title=f"Toss Decisions - {team}"
                        )
                        responsive_plotly_chart(fig, use_container_width=True)
                    else:
                        st.info(f"No toss decision data available for {team}")
                
                with col2:
                    try:
                        success_rate = float(team_stats['success_rate'])
                        st.metric("Success Rate", f"{success_rate:.1f}%")
                    except (TypeError, ValueError, KeyError):
                        st.metric("Success Rate", "N/A")
        else:
            st.info("No team-wise toss preference data available")
    
    # Success Analysis Tab
    with tabs[2]:
        st.subheader("Toss Decision Success Analysis")
        
        # Create success rate comparison
        success_data = pd.DataFrame({
            'Decision': ['Bat First', 'Field First'],
            'Success Rate': [
                stats['decision_success']['bat_first_pct'],
                stats['decision_success']['field_first_pct']
            ]
        })
        
        fig = px.bar(
            success_data,
            x='Decision',
            y='Success Rate',
            title="Success Rate by Toss Decision",
            labels={'Success Rate': 'Win Percentage'}
        )
        responsive_plotly_chart(fig, use_container_width=True)

def display_weather_analysis(matches_df: pd.DataFrame) -> None:
    """Display weather impact analysis for venues."""
    st.header("Weather Analysis")
    
    # Load precomputed weather impact
    venue_weather_impact = load_venue_weather_impact()
    
    if not venue_weather_impact:
        st.error("Weather impact data not available. Please run the data preprocessing script.")
        return
    
    # Get unique venues for selection
    venues = sorted(venue_weather_impact.keys())
    selected_venue = st.selectbox("Select Venue", venues, key="weather_analysis_venue_selector")
    
    if selected_venue not in venue_weather_impact:
        st.warning(f"No weather impact data available for {selected_venue}")
        return
    
    venue_stats = venue_weather_impact[selected_venue]
    
    # Display statistics
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Total Matches", venue_stats['total_matches'])
        st.metric("Rain-Affected Matches", venue_stats['rain_affected_matches'])
        try:
            rain_pct = float(venue_stats['rain_percentage'])
            st.metric("Rain Impact", f"{rain_pct:.1f}%")
        except (TypeError, ValueError, KeyError):
            st.metric("Rain Impact", "N/A")
    
    with col2:
        st.metric("Day Matches", venue_stats['day_matches'])
        st.metric("Evening Matches", venue_stats['evening_matches'])
        try:
            evening_pct = float(venue_stats['evening_percentage'])
            st.metric("Evening Match %", f"{evening_pct:.1f}%")
        except (TypeError, ValueError, KeyError):
            st.metric("Evening Match %", "N/A")
    
    # Create pie chart for match timing distribution
    timing_data = pd.DataFrame({
        'Time': ['Day', 'Evening'],
        'Matches': [venue_stats['day_matches'], venue_stats['evening_matches']]
    })
    
    fig = px.pie(timing_data,
                 values='Matches',
                 names='Time',
                 title=f"Match Timing Distribution at {selected_venue}")
    responsive_plotly_chart(fig)
    
    # Add a note about data availability
    if venue_stats['day_matches'] == venue_stats['total_matches'] // 2 and venue_stats['evening_matches'] == venue_stats['total_matches'] // 2:
        st.info("Note: Match timing data is not available. Showing estimated distribution.")

def display_venue_analysis(matches_df: pd.DataFrame, deliveries_df: pd.DataFrame) -> None:
    """Display comprehensive venue analysis with all enhanced features."""
    st.title("Enhanced Venue Analysis")
    
    # Create tabs for different analyses
    tabs = st.tabs([
        "Venue Overview",
        "Team Performance",
        "Scoring Patterns",
        "Toss Impact",
        "Weather Analysis"
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
    
    # Weather Analysis Tab
    with tabs[4]:
        display_weather_analysis(matches_df) 