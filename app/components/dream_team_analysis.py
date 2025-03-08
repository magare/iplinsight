'''
Module for analyzing Dream Team performance.
This module loads match and dream team data, computes dream team selections, aggregates player statistics,
and creates a Streamlit layout for interactive visualizations.\
'''

import pandas as pd
import plotly.express as px
import streamlit as st
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import os
import json
import numpy as np
from utils.chart_utils import responsive_plotly_chart, update_chart_for_responsive_layout

# Helper functions to load precomputed data
def load_precomputed_json(file_path: Path) -> Dict:
    """
    Load precomputed data from a JSON file.
    
    Args:
        file_path (Path): Path to the JSON file.
        
    Returns:
        Dict: Loaded data or empty dict if file not found.
    """
    try:
        if file_path.exists():
            with open(file_path, 'r') as f:
                return json.load(f)
        else:
            print(f"Warning: Precomputed file not found: {file_path}")
            return {}
    except Exception as e:
        print(f"Error loading precomputed data from {file_path}: {e}")
        return {}

def load_precomputed_parquet(file_path: Path) -> Dict:
    """
    Load precomputed data from a Parquet file.
    
    Args:
        file_path (Path): Path to the Parquet file.
        
    Returns:
        Dict: Loaded data or empty dict if file not found.
    """
    # Convert file_path from .json to .parquet if needed
    if str(file_path).endswith('.json'):
        parquet_path = Path(str(file_path).replace('.json', '.parquet'))
    else:
        parquet_path = file_path
    
    try:
        if parquet_path.exists():
            df = pd.read_parquet(parquet_path)
            # Convert DataFrame to dict
            if len(df) == 1:
                # If it's a single row DataFrame (converted from a dict)
                return df.iloc[0].to_dict()
            else:
                # If it's a multi-row DataFrame (converted from a list)
                return df.to_dict(orient='records')
        else:
            # Try to fall back to JSON if Parquet doesn't exist
            if file_path.exists():
                return load_precomputed_json(file_path)
            print(f"Warning: Precomputed file not found: {parquet_path}")
            return {}
    except Exception as e:
        print(f"Error loading precomputed data from {parquet_path}: {e}")
        # Try to fall back to JSON if Parquet loading fails
        if file_path.exists():
            return load_precomputed_json(file_path)
        return {}

def load_dream_team_all_time_stats() -> pd.DataFrame:
    """
    Load precomputed all-time dream team statistics.
    
    Returns:
        pd.DataFrame: DataFrame containing all-time player statistics.
    """
    data_dir = Path(__file__).resolve().parent.parent / 'data'
    file_path = data_dir / 'dream_team_all_time_stats.json'
    
    # Try to load from Parquet first
    parquet_path = file_path.with_suffix('.parquet')
    if parquet_path.exists():
        try:
            # The Parquet file might have a different structure
            df = pd.read_parquet(parquet_path)
            
            # If it's a DataFrame with a single column containing the data
            if len(df.columns) == 1 and isinstance(df.iloc[0, 0], (list, np.ndarray)):
                return pd.DataFrame(df.iloc[0, 0])
            
            # If it's already a proper DataFrame with player data
            elif 'player' in df.columns:
                return df
            
            # If it's a DataFrame with a single row containing the data
            elif len(df) == 1:
                for col in df.columns:
                    if isinstance(df.iloc[0][col], (list, np.ndarray)):
                        return pd.DataFrame(df.iloc[0][col])
            
            return df
        except Exception as e:
            print(f"Error loading Parquet file: {e}")
            # Fall back to JSON
    
    # Fall back to JSON if Parquet loading fails
    if file_path.exists():
        data = load_precomputed_json(file_path)
        if data:
            return pd.DataFrame(data)
    
    return pd.DataFrame()

def load_dream_team_season_stats(season: int) -> pd.DataFrame:
    """
    Load precomputed season-specific dream team statistics.
    
    Args:
        season (int): IPL season year.
        
    Returns:
        pd.DataFrame: DataFrame containing season-specific player statistics.
    """
    data_dir = Path(__file__).resolve().parent.parent / 'data'
    file_path = data_dir / 'dream_team_season_stats.json'
    
    # Try to load from Parquet first
    parquet_path = file_path.with_suffix('.parquet')
    if parquet_path.exists():
        try:
            # The Parquet file has a different structure - it's a DataFrame with season columns
            df = pd.read_parquet(parquet_path)
            if str(season) in df.columns:
                # Extract the array from the first row of the season column
                season_data = df.iloc[0][str(season)]
                if isinstance(season_data, (list, np.ndarray)):
                    return pd.DataFrame(season_data)
            return pd.DataFrame()
        except Exception as e:
            print(f"Error loading Parquet file: {e}")
            # Fall back to JSON
    
    # Fall back to JSON if Parquet loading fails
    if file_path.exists():
        data = load_precomputed_json(file_path)
        if data and str(season) in data:
            return pd.DataFrame(data[str(season)])
    
    return pd.DataFrame()

def load_dream_team_venue_stats(venue: str) -> pd.DataFrame:
    """
    Load precomputed venue-specific dream team statistics.
    
    Args:
        venue (str): Stadium/venue name.
        
    Returns:
        pd.DataFrame: DataFrame containing venue-specific player statistics.
    """
    data_dir = Path(__file__).resolve().parent.parent / 'data'
    file_path = data_dir / 'dream_team_venue_stats.json'
    
    # Try to load from Parquet first
    parquet_path = file_path.with_suffix('.parquet')
    if parquet_path.exists():
        try:
            # The Parquet file has a different structure - it's a DataFrame with venue columns or rows
            df = pd.read_parquet(parquet_path)
            
            # Check if venues are columns
            if venue in df.columns:
                venue_data = df.iloc[0][venue]
                if isinstance(venue_data, (list, np.ndarray)):
                    return pd.DataFrame(venue_data)
            
            # Check if venues are in a 'venue' column
            elif 'venue' in df.columns:
                venue_data = df[df['venue'] == venue]
                if not venue_data.empty:
                    return venue_data
                
            return pd.DataFrame()
        except Exception as e:
            print(f"Error loading Parquet file: {e}")
            # Fall back to JSON
    
    # Fall back to JSON if Parquet loading fails
    if file_path.exists():
        data = load_precomputed_json(file_path)
        if data and venue in data:
            return pd.DataFrame(data[venue])
    
    return pd.DataFrame()

def load_dream_team_player_history(player: str) -> pd.DataFrame:
    """
    Load precomputed player history for dream team.
    
    Args:
        player (str): Player name.
        
    Returns:
        pd.DataFrame: DataFrame containing player's match history.
    """
    data_dir = Path(__file__).resolve().parent.parent / 'data'
    file_path = data_dir / 'dream_team_player_history.json'
    
    # Try to load from Parquet first
    parquet_path = file_path.with_suffix('.parquet')
    if parquet_path.exists():
        try:
            # The Parquet file has a different structure - it's a DataFrame with player columns or rows
            df = pd.read_parquet(parquet_path)
            
            # Check if players are columns
            if player in df.columns:
                player_data = df.iloc[0][player]
                if isinstance(player_data, (list, np.ndarray)):
                    return pd.DataFrame(player_data)
            
            # Check if players are in a 'player' column
            elif 'player' in df.columns:
                player_data = df[df['player'] == player]
                if not player_data.empty:
                    return player_data
                
            return pd.DataFrame()
        except Exception as e:
            print(f"Error loading Parquet file: {e}")
            # Fall back to JSON
    
    # Fall back to JSON if Parquet loading fails
    if file_path.exists():
        data = load_precomputed_json(file_path)
        if data and player in data:
            return pd.DataFrame(data[player])
    
    return pd.DataFrame()

def load_dream_team_match_team(match_id: str) -> pd.DataFrame:
    """
    Load precomputed dream team for a specific match.
    
    Args:
        match_id (str): Match ID.
        
    Returns:
        pd.DataFrame: DataFrame containing dream team for the match.
    """
    data_dir = Path(__file__).resolve().parent.parent / 'data'
    file_path = data_dir / 'dream_team_match_teams.json'
    
    # Try to load from Parquet first
    parquet_path = file_path.with_suffix('.parquet')
    if parquet_path.exists():
        try:
            # The Parquet file has a different structure - it's a DataFrame with match_id columns or rows
            df = pd.read_parquet(parquet_path)
            
            # Check if match_ids are columns
            if match_id in df.columns:
                match_data = df.iloc[0][match_id]
                if isinstance(match_data, (list, np.ndarray)):
                    return pd.DataFrame(match_data)
            
            # Check if match_ids are in a 'match_id' column
            elif 'match_id' in df.columns:
                match_data = df[df['match_id'] == match_id]
                if not match_data.empty:
                    return match_data
                
            return pd.DataFrame()
        except Exception as e:
            print(f"Error loading Parquet file: {e}")
            # Fall back to JSON
    
    # Fall back to JSON if Parquet loading fails
    if file_path.exists():
        data = load_precomputed_json(file_path)
        if data and str(match_id) in data:
            return pd.DataFrame(data[str(match_id)])
    
    return pd.DataFrame()

def load_dream_team_all_matches() -> Dict[int, List[Dict]]:
    """
    Load precomputed dream teams for all matches.
    
    Returns:
        Dict[int, List[Dict]]: Dictionary mapping seasons to lists of match dream teams.
    """
    data_dir = Path(__file__).resolve().parent.parent / 'data'
    file_path = data_dir / 'dream_team_all_matches.json'
    
    # Try to load from Parquet first
    parquet_path = file_path.with_suffix('.parquet')
    if parquet_path.exists():
        try:
            # The Parquet file has a different structure - it's a DataFrame with season columns
            df = pd.read_parquet(parquet_path)
            
            # Convert DataFrame back to dictionary
            result = {}
            for col in df.columns:
                if isinstance(df.iloc[0][col], (list, np.ndarray)):
                    result[col] = df.iloc[0][col].tolist()
            
            if result:
                return result
        except Exception as e:
            print(f"Error loading Parquet file: {e}")
            # Fall back to JSON
    
    # Fall back to JSON if Parquet loading fails
    if file_path.exists():
        return load_precomputed_json(file_path)
    
    return {}

@st.cache_data
def load_dream_team_data() -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Dict]]:
    """
    Load match and dream team data from CSV files.
    
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Dict]]: A tuple containing:
            - matches_df: DataFrame with match information
            - dream_team_df: DataFrame with dream team statistics
            - match_info: Dictionary mapping match_id to match details
    """
    data_dir = Path(__file__).resolve().parent.parent / 'data'
    
    # Load match data
    matches_df = pd.read_parquet(data_dir / 'matches.parquet')
    
    # Load dream team data
    dream_team_df = pd.read_parquet(data_dir / 'dream_team_stats.parquet')
    
    # Create match info dictionary for quick lookup
    match_info = {}
    for _, row in matches_df.iterrows():
        match_info[str(row['match_id'])] = {
            'season': row['season'],
            'date': row['date'],
            'venue': row['venue'],
            'team1': row['team1'],
            'team2': row['team2'],
            'winner': row['winner'],
            'toss_winner': row['toss_winner'],
            'toss_decision': row['toss_decision'],
            'player_of_match': row['player_of_match']
        }
    
    return matches_df, dream_team_df, match_info


class DreamTeamAnalysis:
    """
    Class for performing Dream Team analysis including generating reports and visualizations.
    """
    def __init__(self):
        """
        Initializes the DreamTeamAnalysis class by loading the required data.
        """
        # Get the path to the app/data directory
        current_dir = Path(__file__).resolve().parent
        data_dir = current_dir.parent / 'data'
        
        # Load data
        self.matches_df, self.dream_team_df, self.match_info = load_dream_team_data()
    
    def get_match_dream_team(self, match_id: str) -> pd.DataFrame:
        """
        Retrieves the dream team for a specific match.

        Args:
            match_id (str): Identifier for the match.

        Returns:
            pd.DataFrame: Dream team for the given match.
        """
        # Try to load from precomputed data first
        dream_team = load_dream_team_match_team(match_id)
        if not dream_team.empty:
            return dream_team
        
        # Fall back to computing on the fly if precomputed data is not available
        return cached_get_match_dream_team(match_id, self.dream_team_df)

    def get_all_time_dream_team_stats(self) -> pd.DataFrame:
        """
        Retrieves aggregated dream team statistics across all matches.

        Returns:
            pd.DataFrame: Aggregated all-time player statistics.
        """
        # Try to load from precomputed data first
        all_time_stats = load_dream_team_all_time_stats()
        if not all_time_stats.empty:
            return all_time_stats
        
        # Fall back to computing on the fly if precomputed data is not available
        return cached_all_time_stats(self.dream_team_df)

    def get_season_dream_team_stats(self, season: int) -> pd.DataFrame:
        """
        Retrieves dream team statistics specific to a given season.

        Args:
            season (int): Season year identifier.

        Returns:
            pd.DataFrame: Aggregated season-specific player statistics.
        """
        # Try to load from precomputed data first
        season_stats = load_dream_team_season_stats(season)
        if not season_stats.empty:
            return season_stats
        
        # Fall back to computing on the fly if precomputed data is not available
        return cached_season_stats(season, self.matches_df, self.dream_team_df)

    def get_venue_dream_team_stats(self, venue: str) -> pd.DataFrame:
        """
        Retrieves dream team statistics for a specified venue.

        Args:
            venue (str): Venue name.

        Returns:
            pd.DataFrame: Aggregated venue-specific player statistics.
        """
        # Try to load from precomputed data first
        venue_stats = load_dream_team_venue_stats(venue)
        if not venue_stats.empty:
            return venue_stats
        
        # Fall back to computing on the fly if precomputed data is not available
        return cached_venue_stats(venue, self.matches_df, self.dream_team_df)

    def get_player_dream_team_history(self, player: str) -> pd.DataFrame:
        """
        Retrieves the history of dream team appearances for a specific player.

        Args:
            player (str): Player's name.

        Returns:
            pd.DataFrame: DataFrame containing career history of dream team appearances for the player.
        """
        # Try to load from precomputed data first
        player_history = load_dream_team_player_history(player)
        if not player_history.empty:
            return player_history
        
        # Fall back to computing on the fly if precomputed data is not available
        return cached_player_history(player, self.dream_team_df, self.matches_df)

    def get_all_matches_dream_teams(self) -> Dict[int, List[Dict]]:
        """
        Retrieves dream teams for all matches, organized by season.

        Returns:
            Dict[int, List[Dict]]: Dictionary mapping season to a list of match entries, each containing
                                   match info and corresponding dream team.
        """
        # Try to load from precomputed data first
        all_matches = load_dream_team_all_matches()
        if all_matches:
            # Convert season keys to integers for consistency
            return {int(k): v for k, v in all_matches.items()}
        
        # Fall back to computing on the fly if precomputed data is not available
        return cached_all_matches_dream_teams(self.dream_team_df, self.match_info)

    def create_layout(self):
        """
        Creates and renders the Streamlit layout for the Dream Team Analysis page.
        Organizes multiple tabs that display various statistics and charts.
        """
        st.header("Dream Team Analysis")
        tabs = st.tabs([
            "Overall Landscape",
            "Season Analysis",
            "Match Analysis",
            "Venue Analysis",
            "Player Profiles",
            "All Match Dream Teams"
        ])

        # --- Overall Landscape ---
        with tabs[0]:
            st.subheader("Overall Dream Team Landscape")
            with st.expander("Dream Team Selection Rules"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown("### Batting Points")
                    st.markdown("""
                    - Run: +1  
                    - Boundary Bonus: +1  
                    - Six Bonus: +2  
                    - 30 Run Bonus: +4  
                    - Half-Century Bonus: +8  
                    - Century Bonus: +16  
                    - Dismissal for duck: -2
                    """)
                    st.markdown("### Strike Rate Points")
                    st.markdown("""
                    (Min 10 Balls)  
                    - Above 170: +6  
                    - 150.01-170: +4  
                    - 130-150: +2  
                    - 60-70: -2  
                    - 50-59.99: -4  
                    - Below 50: -6
                    """)
                with col2:
                    st.markdown("### Bowling Points")
                    st.markdown("""
                    - Wicket (Excluding Run Out): +25  
                    - Bonus (LBW/Bowled): +8  
                    - 3 Wicket Bonus: +4  
                    - 4 Wicket Bonus: +8  
                    - 5 Wicket Bonus: +16  
                    - Maiden Over: +12
                    """)
                    st.markdown("### Economy Rate Points")
                    st.markdown("""
                    (Min 2 Overs)  
                    - Below 5: +6  
                    - 5-5.99: +4  
                    - 6-7: +2  
                    - 10-11: -2  
                    - 11.01-12: -4  
                    - Above 12: -6
                    """)
                with col3:
                    st.markdown("### Fielding Points")
                    st.markdown("""
                    - Catch: +8  
                    - 3 Catch Bonus: +4  
                    - Stumping: +12  
                    - Run out (Not direct): +6
                    """)
                    st.markdown("### Other Points")
                    st.markdown("""
                    - Captain: 2X  
                    - Vice Captain: 1.5X
                    """)
            all_time_stats = self.get_all_time_dream_team_stats()
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Top Players by Appearances")
                fig = px.bar(
                    all_time_stats.nlargest(10, 'appearances'),
                    x='player', y='appearances',
                    color='primary_role',
                    title='Top 10 Players by Dream Team Appearances'
                )
                responsive_plotly_chart(fig)
            with col2:
                st.subheader("Top Players by Appearance Rate")
                fig = px.bar(
                    all_time_stats.nlargest(10, 'appearance_rate'),
                    x='player', y='appearance_rate',
                    color='primary_role',
                    title='Top 10 Players by Dream Team Appearance Rate'
                )
                responsive_plotly_chart(fig)
            st.subheader("Overall Dream Team Composition")
            role_comp = all_time_stats['primary_role'].value_counts()
            fig = px.pie(values=role_comp.values, names=role_comp.index,
                         title='Dream Team Role Distribution')
            responsive_plotly_chart(fig)
            st.subheader("Top All-Time Performers")
            metrics_df = all_time_stats.nlargest(20, 'appearances').copy()
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("### Top Run Scorers")
                st.dataframe(metrics_df[['player', 'total_runs', 'appearances']]
                            .sort_values('total_runs', ascending=False).head(10).reset_index(drop=True).assign(index=lambda x: x.index + 1).set_index('index'))
            with col2:
                st.markdown("### Top Wicket Takers")
                st.dataframe(metrics_df[['player', 'total_wickets', 'appearances']]
                            .sort_values('total_wickets', ascending=False).head(10).reset_index(drop=True).assign(index=lambda x: x.index + 1).set_index('index'))
            with col3:
                st.markdown("### Best Fielders")
                metrics_df['total_dismissals'] = (metrics_df['total_catches'] +
                                                metrics_df['total_stumpings'] +
                                                metrics_df['total_run_outs'])
                st.dataframe(metrics_df[['player', 'total_dismissals', 'appearances']]
                            .sort_values('total_dismissals', ascending=False).head(10).reset_index(drop=True).assign(index=lambda x: x.index + 1).set_index('index'))
        
        # --- Season Analysis ---
        with tabs[1]:
            st.subheader("Season-Specific Dream Teams")
            selected_season = st.selectbox("Select Season",
                                           options=sorted(self.matches_df['season'].unique()))
            if selected_season:
                season_stats = self.get_season_dream_team_stats(selected_season)
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader(f"Top Players in {selected_season}")
                    fig = px.bar(
                        season_stats.nlargest(10, 'appearances'),
                        x='player', y='appearances',
                        color='primary_role',
                        title=f'Top 10 Players by Dream Team Appearances in {selected_season}'
                    )
                    responsive_plotly_chart(fig)
                with col2:
                    st.subheader("Role Distribution")
                    role_comp = season_stats['primary_role'].value_counts()
                    fig = px.pie(values=role_comp.values, names=role_comp.index,
                                 title=f'Dream Team Role Distribution in {selected_season}')
                    responsive_plotly_chart(fig)
                st.subheader(f"Season {selected_season} Best XI")
                best_xi = season_stats.nlargest(11, 'appearances')
                st.dataframe(best_xi[['player', 'primary_role', 'appearances', 'avg_points',
                                      'total_runs', 'total_wickets', 'total_catches']]
                             .reset_index(drop=True)
                             .assign(index=lambda x: x.index + 1)
                             .set_index('index'), 
                             use_container_width=True)
                st.subheader("Season Performance Analysis")
                col1, col2, col3 = st.columns(3)
                with col1:
                    try:
                        max_appearances = int(best_xi['appearances'].max())
                        player_name = best_xi.iloc[0]['player']
                        st.metric("Most Dream Team Appearances",
                                f"{max_appearances}",
                                f"by {player_name}")
                    except (TypeError, ValueError, KeyError, AttributeError, IndexError):
                        st.metric("Most Dream Team Appearances", "N/A")
                with col2:
                    try:
                        max_avg_points = float(best_xi['avg_points'].max())
                        max_points_player = best_xi.loc[best_xi['avg_points'].idxmax(), 'player']
                        st.metric("Highest Average Points",
                                f"{max_avg_points:.1f}",
                                f"by {max_points_player}")
                    except (TypeError, ValueError, KeyError, AttributeError):
                        st.metric("Highest Average Points", "N/A")
                with col3:
                    st.metric("Most Consistent Role",
                              best_xi['primary_role'].mode().iloc[0],
                              f"{(best_xi['primary_role'] == best_xi['primary_role'].mode().iloc[0]).sum()} players")
        
        # --- Match Analysis ---
        with tabs[2]:
            st.subheader("Match-Level Dream Team Insights")
            match_id = st.selectbox("Select Match",
                                     options=sorted(self.matches_df['match_id'].unique()),
                                     format_func=lambda x: f"Match {x}")
            if match_id:
                dream_team_df = self.get_match_dream_team(str(match_id))
                match_info = self.matches_df[self.matches_df['match_id'] == match_id].iloc[0]
                st.markdown(f"""
                **Match Details:**  
                - Season: {match_info['season']}  
                - Teams: {match_info['team1']} vs {match_info['team2']}  
                - Venue: {match_info['venue']}  
                - Winner: {match_info['winner']}
                """)
                st.markdown("### Dream Team Players")
                st.dataframe(dream_team_df[[
                    'player', 'role', 'captain_role', 'total_points',
                    'batting_points', 'bowling_points', 'fielding_points'
                ]].reset_index(drop=True).assign(index=lambda x: x.index + 1).set_index('index').style.format({
                    'total_points': '{:.2f}',
                    'batting_points': '{:.2f}',
                    'bowling_points': '{:.2f}',
                    'fielding_points': '{:.2f}'
                }), use_container_width=True)
                fig = px.bar(
                    dream_team_df,
                    x='player',
                    y=['batting_points', 'bowling_points', 'fielding_points'],
                    title='Player Points Breakdown',
                    labels={'value': 'Points', 'variable': 'Category'},
                    barmode='stack'
                )
                responsive_plotly_chart(fig)
                
                # Create side-by-side tables for batting and bowling performances
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Batting Performances")
                    batting_stats = dream_team_df[['player', 'runs', 'boundaries', 'sixes', 'strike_rate']].sort_values('runs', ascending=False)
                    st.dataframe(batting_stats.reset_index(drop=True).assign(index=lambda x: x.index + 1).set_index('index'), use_container_width=True)
                
                with col2:
                    st.subheader("Bowling Performances")
                    bowling_stats = dream_team_df[['player', 'wickets', 'maidens', 'economy_rate']].sort_values('wickets', ascending=False)
                    st.dataframe(bowling_stats.reset_index(drop=True).assign(index=lambda x: x.index + 1).set_index('index'), use_container_width=True)
        
        # --- Venue Analysis ---
        with tabs[3]:
            st.subheader("Venue-Specific Dream Teams")
            selected_venue = st.selectbox("Select Venue",
                                          options=sorted(self.matches_df['venue'].unique()))
            if selected_venue:
                venue_stats = self.get_venue_dream_team_stats(selected_venue)
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader(f"Top Players at {selected_venue}")
                    fig = px.bar(
                        venue_stats.nlargest(10, 'appearances'),
                        x='player', y='appearances',
                        color='primary_role',
                        title=f'Top 10 Players by Dream Team Appearances at {selected_venue}'
                    )
                    responsive_plotly_chart(fig)
                with col2:
                    st.subheader("Role Distribution")
                    role_comp = venue_stats['primary_role'].value_counts()
                    fig = px.pie(values=role_comp.values, names=role_comp.index,
                                 title=f'Dream Team Role Distribution at {selected_venue}')
                    responsive_plotly_chart(fig)
                st.subheader("Venue Best XI")
                best_xi = venue_stats.nlargest(11, 'appearances')
                st.dataframe(best_xi[['player', 'primary_role', 'appearances', 'avg_points',
                                      'total_runs', 'total_wickets', 'total_catches']]
                             .reset_index(drop=True)
                             .assign(index=lambda x: x.index + 1)
                             .set_index('index'), 
                             use_container_width=True)
                st.subheader("Venue Statistics")
                total_matches = self.matches_df[self.matches_df['venue'] == selected_venue].shape[0]
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Matches", total_matches)
                with col2:
                    st.metric("Most Appearances",
                              f"{best_xi['appearances'].max():.0f}",
                              f"by {best_xi.iloc[0]['player']}")
                with col3:
                    st.metric("Dominant Role",
                              best_xi['primary_role'].mode().iloc[0],
                              f"{(best_xi['primary_role'] == best_xi['primary_role'].mode().iloc[0]).sum()} players")
        
        # --- Player Profiles ---
        with tabs[4]:
            st.subheader("Player Dream Team Profiles")
            all_time_stats = self.get_all_time_dream_team_stats()
            selected_player = st.selectbox("Select Player",
                                           options=sorted(all_time_stats['player'].unique()))
            if selected_player:
                player_history = self.get_player_dream_team_history(selected_player)
                player_stats = all_time_stats[all_time_stats['player'] == selected_player].iloc[0]
                if not player_history.empty:
                    st.markdown(f"""
                    ### Overall Dream Team Stats for {selected_player}  
                    - Total Appearances: {player_stats['appearances']}  
                    - Matches Played: {player_stats['matches_played']}  
                    - Appearance Rate: {player_stats['appearance_rate']:.2%}  
                    - Average Points: {player_stats['avg_points']:.2f}  
                    - Primary Role: {player_stats['primary_role']}
                    """)
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Runs", f"{player_stats['total_runs']:.0f}")
                    with col2:
                        st.metric("Total Wickets", f"{player_stats['total_wickets']:.0f}")
                    with col3:
                        total_dismissals = player_stats['total_catches'] + player_stats['total_stumpings'] + player_stats['total_run_outs']
                        st.metric("Total Dismissals", f"{total_dismissals:.0f}")
                    st.subheader("Performance Trend")
                    fig = px.line(
                        player_history.sort_values('date'),
                        x='date', y='total_points',
                        title=f'{selected_player} Dream Team Performance Trend',
                        labels={'date': 'Date', 'total_points': 'Fantasy Points'}
                    )
                    responsive_plotly_chart(fig)
                    st.subheader("Season Breakdown")
                    season_breakdown = player_history.groupby('season').agg(
                        Appearances=('match_id', 'count'),
                        **{'Avg Points': ('total_points', 'mean'),
                           'Total Runs': ('runs', 'sum'),
                           'Total Wickets': ('wickets', 'sum'),
                           'Total Catches': ('catches', 'sum')}
                    )
                    st.dataframe(season_breakdown.reset_index().assign(Season=lambda x: x['season']).set_index('Season'))
                    st.subheader("Venue Breakdown")
                    venue_breakdown = player_history.groupby('venue').agg(
                        Appearances=('match_id', 'count'),
                        **{'Avg Points': ('total_points', 'mean'),
                           'Total Runs': ('runs', 'sum'),
                           'Total Wickets': ('wickets', 'sum'),
                           'Total Catches': ('catches', 'sum')}
                    )
                    st.dataframe(venue_breakdown.reset_index().set_index('venue'))
                    st.subheader("Recent Form")
                    recent_matches = player_history.sort_values('date', ascending=False).head(5)
                    st.dataframe(recent_matches[['date', 'teams', 'total_points', 'runs', 'wickets', 'catches']].reset_index(drop=True).assign(index=lambda x: x.index + 1).set_index('index'))
                else:
                    st.warning(f"No dream team appearances found for {selected_player}")
        
        # --- All Match Dream Teams ---
        with tabs[5]:
            st.subheader("Dream Teams for All Matches")
            match_ids = self.matches_df['match_id'].unique()
            for match_id in match_ids:
                with st.expander(f"Match ID: {match_id}"):
                    if st.button(f"Load Dream Team for Match {match_id}"):
                        dream_team = self.get_match_dream_team(match_id)
                        if not dream_team.empty:
                            st.dataframe(dream_team.reset_index(drop=True).assign(index=lambda x: x.index + 1).set_index('index'))
                        else:
                            st.write("No data available for this match.")

    def display_dream_team_visualization(self, dream_team, season, match_id, idx):
        """Display visualizations for the dream team."""
        col1, col2 = st.columns(2)
        
        with col1:
            # Check if dream_team is a DataFrame before creating the bar chart
            if isinstance(dream_team, pd.DataFrame) and not dream_team.empty and 'player' in dream_team.columns:
                # Check if all required columns exist
                point_columns = ['batting_points', 'bowling_points', 'fielding_points']
                available_point_columns = [col for col in point_columns if col in dream_team.columns]
                
                if available_point_columns:
                    fig = px.bar(
                        dream_team,
                        x='player',
                        y=available_point_columns,
                        title='Points Distribution',
                        labels={'value': 'Points', 'variable': 'Category'},
                        barmode='stack',
                        height=300
                    )
                    fig.update_layout(showlegend=False, margin=dict(t=30, b=0))
                    # Replace with responsive chart
                    responsive_plotly_chart(fig, use_container_width=True,
                                        key=f"points_dist_{season}_{str(match_id)}_{idx}")
                else:
                    st.warning("Points data not available for visualization")
            else:
                st.warning("Player data not available for visualization")
            
        with col2:
            # Check if dream_team is a DataFrame before calculating role counts
            if isinstance(dream_team, pd.DataFrame) and not dream_team.empty and 'role' in dream_team.columns:
                role_counts = dream_team['role'].value_counts()
                if not role_counts.empty:
                    fig = px.pie(
                        values=role_counts.values,
                        names=role_counts.index,
                        title='Team Composition',
                        height=300
                    )
                    fig.update_layout(showlegend=True, margin=dict(t=30, b=0))
                    # Replace with responsive chart
                    responsive_plotly_chart(fig, use_container_width=True,
                                         key=f"role_dist_{season}_{str(match_id)}_{idx}")
                else:
                    st.warning("No role data available for visualization")
            else:
                st.warning("Role data not available for visualization")

    @staticmethod
    def _compute_dream_team(df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes the dream team for a match by selecting the top 11 players based on total points,
        and assigns Captain and Vice-Captain roles with corresponding point multipliers.
        
        Args:
            df (pd.DataFrame): DataFrame containing player performance data for a match.
        
        Returns:
            pd.DataFrame: DataFrame representing the computed dream team.
        """
        # Check if input is a DataFrame and not empty
        if not isinstance(df, pd.DataFrame) or df.empty:
            return pd.DataFrame()
            
        # Ensure 'total_points' column exists
        if 'total_points' not in df.columns:
            return pd.DataFrame()
            
        try:
            dream_team = df.nlargest(11, 'total_points').copy()
            
            # Ensure 'captain_role' column exists or create it
            if 'captain_role' not in dream_team.columns:
                dream_team['captain_role'] = 'Player'
            else:
                dream_team['captain_role'] = 'Player'
                
            if len(dream_team) > 0:
                dream_team.iloc[0, dream_team.columns.get_loc('captain_role')] = 'Captain'
            if len(dream_team) > 1:
                dream_team.iloc[1, dream_team.columns.get_loc('captain_role')] = 'Vice-Captain'
                
            # Apply multipliers: double points for Captain, 1.5x for Vice-Captain
            if 'total_points' in dream_team.columns:
                dream_team.loc[dream_team['captain_role'] == 'Captain', 'total_points'] *= 2
                dream_team.loc[dream_team['captain_role'] == 'Vice-Captain', 'total_points'] *= 1.5
                
            return dream_team
        except Exception as e:
            # If any error occurs, return an empty DataFrame
            print(f"Error computing dream team: {str(e)}")
            return pd.DataFrame()

    @staticmethod
    def _aggregate_stats(df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregates player statistics from dream team data.
        
        Args:
            df (pd.DataFrame): DataFrame containing dream team selections.
        
        Returns:
            pd.DataFrame: Aggregated statistics including matches played, total and average points,
                        runs, wickets, fielding metrics, and role information. 
                        If the DataFrame is empty, returns an empty DataFrame with predefined columns.
        """
        if df.empty:
            return pd.DataFrame(columns=[
                'player', 'matches_played', 'total_points', 'avg_points', 
                'total_runs', 'total_wickets', 'total_catches', 'total_stumpings', 
                'total_run_outs', 'appearances', 'primary_role', 'appearance_rate'
            ])
        grouped = df.groupby('player').agg(
            matches_played=('match_id', 'nunique'),
            total_points=('total_points', 'sum'),
            avg_points=('total_points', 'mean'),
            total_runs=('runs', 'sum'),
            total_wickets=('wickets', 'sum'),
            total_catches=('catches', 'sum'),
            total_stumpings=('stumpings', 'sum'),
            total_run_outs=('run_outs', 'sum')
        ).reset_index()
        grouped['appearances'] = grouped['matches_played']
        mode_roles = df.groupby('player')['role'].agg(lambda x: x.mode().iat[0]).reset_index(name='primary_role')
        stats = pd.merge(grouped, mode_roles, on='player')
        stats['appearance_rate'] = stats['appearances'] / stats['matches_played']
        return stats


# -------------------------------
# Module-level cached functions
# These functions utilize caching for performance and take only hashable arguments.
# -------------------------------

@st.cache_data(show_spinner=False)
def cached_get_match_dream_team(match_id: str, dream_team_df: pd.DataFrame) -> pd.DataFrame:
    """
    Cached helper function to compute a match-specific dream team.
    
    Args:
        match_id (str): Identifier of the match.
        dream_team_df (pd.DataFrame): DataFrame containing dream team data.
    
    Returns:
        pd.DataFrame: Computed dream team for the match.
    """
    df = dream_team_df[dream_team_df['match_id'] == str(match_id)]
    return DreamTeamAnalysis._compute_dream_team(df)


@st.cache_data(show_spinner=False)
def cached_all_time_stats(dream_team_df: pd.DataFrame) -> pd.DataFrame:
    """
    Cached helper function to aggregate all-time dream team statistics.
    
    Args:
        dream_team_df (pd.DataFrame): DataFrame containing dream team data.
    
    Returns:
        pd.DataFrame: Aggregated all-time player statistics.
    """
    return DreamTeamAnalysis._aggregate_stats(dream_team_df)


@st.cache_data(show_spinner=False)
def cached_season_stats(season: int, matches_df: pd.DataFrame, dream_team_df: pd.DataFrame) -> pd.DataFrame:
    """
    Cached helper function to aggregate season-specific dream team statistics.
    
    Args:
        season (int): Season year.
        matches_df (pd.DataFrame): DataFrame containing match data.
        dream_team_df (pd.DataFrame): DataFrame containing dream team data.
    
    Returns:
        pd.DataFrame: Aggregated season-specific player statistics.
    """
    season_matches = matches_df[matches_df['season'] == season]['match_id'].unique()
    season_stats = dream_team_df[dream_team_df['match_id'].isin(season_matches)]
    return DreamTeamAnalysis._aggregate_stats(season_stats)


@st.cache_data(show_spinner=False)
def cached_venue_stats(venue: str, matches_df: pd.DataFrame, dream_team_df: pd.DataFrame) -> pd.DataFrame:
    """
    Cached helper function to aggregate venue-specific dream team statistics.
    
    Args:
        venue (str): Venue name.
        matches_df (pd.DataFrame): DataFrame containing match data.
        dream_team_df (pd.DataFrame): DataFrame containing dream team data.
    
    Returns:
        pd.DataFrame: Aggregated venue-specific player statistics.
    """
    venue_matches = matches_df[matches_df['venue'] == venue]['match_id'].unique()
    venue_stats = dream_team_df[dream_team_df['match_id'].isin(venue_matches)]
    return DreamTeamAnalysis._aggregate_stats(venue_stats)


@st.cache_data(show_spinner=False)
def cached_player_history(player: str, dream_team_df: pd.DataFrame, matches_df: pd.DataFrame) -> pd.DataFrame:
    """
    Cached helper function to retrieve a player's dream team appearance history.
    
    Args:
        player (str): Player's name.
        dream_team_df (pd.DataFrame): DataFrame containing dream team data.
        matches_df (pd.DataFrame): DataFrame containing match data.
    
    Returns:
        pd.DataFrame: DataFrame containing the player's history of dream team appearances.
    """
    player_stats = dream_team_df[dream_team_df['player'] == player].copy()
    if player_stats.empty:
        return player_stats
    merge_cols = ['match_id', 'season', 'date', 'venue', 'team1', 'team2', 'winner']
    player_stats = player_stats.merge(matches_df[merge_cols], on='match_id', how='left')
    player_stats['teams'] = player_stats['team1'] + " vs " + player_stats['team2']
    return player_stats


@st.cache_data(show_spinner=False)
def cached_all_matches_dream_teams(dream_team_df: pd.DataFrame, match_info: dict) -> Dict[int, List[Dict]]:
    """
    Cached helper function to group dream teams by match and season.
    
    Args:
        dream_team_df (pd.DataFrame): DataFrame containing dream team data.
        match_info (dict): Dictionary mapping match_id to match details.
    
    Returns:
        Dict[int, List[Dict]]: Dictionary where keys are seasons and values are lists of match entries, each containing
                               match info and corresponding dream team.
    """
    grouped = dream_team_df.groupby('match_id')
    match_dreams = {}
    
    # Safely compute dream team for each match
    for mid, group in grouped:
        try:
            match_dreams[str(mid)] = DreamTeamAnalysis._compute_dream_team(group)
        except Exception as e:
            st.warning(f"Error computing dream team for match {mid}: {str(e)}")
            # Ensure we return an empty DataFrame, not None or another type
            match_dreams[str(mid)] = pd.DataFrame()
    
    seasons = {}
    for match_id, dream_team in match_dreams.items():
        if match_id in match_info:
            # Ensure dream_team is a DataFrame
            if not isinstance(dream_team, pd.DataFrame):
                dream_team = pd.DataFrame()
                
            season = match_info[match_id]['season']
            match_info_dict = match_info[match_id].copy()
            match_info_dict['match_id'] = match_id  # Ensure match_id is included
            entry = {'match_info': match_info_dict, 'dream_team': dream_team}
            seasons.setdefault(season, []).append(entry)
    return seasons

# Extract the data range from matches_df
min_year = load_dream_team_data()[0]['season'].min()
max_year = load_dream_team_data()[0]['season'].max()

# Update the sidebar label
st.sidebar.markdown("---")
st.sidebar.markdown(f"""
<div style="opacity: 0.7; font-size: 0.8rem; text-align: center;">
    <p>Data from {min_year}-{max_year} IPL seasons</p>
</div>
""", unsafe_allow_html=True)
