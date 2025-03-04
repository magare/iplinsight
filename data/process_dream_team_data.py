import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from pathlib import Path
import os

def calculate_batting_points(player_deliveries: pd.DataFrame) -> Dict[str, float]:
    """Calculate batting points for a player in a match"""
    if player_deliveries.empty:
        return {
            'total_points': 0.0,
            'runs': 0,
            'boundaries': 0,
            'sixes': 0,
            'strike_rate': 0.0,
            'balls_faced': 0
        }
    
    total_runs = player_deliveries['batsman_runs'].sum()
    boundaries = len(player_deliveries[player_deliveries['batsman_runs'] == 4])
    sixes = len(player_deliveries[player_deliveries['batsman_runs'] == 6])
    balls_faced = len(player_deliveries)
    strike_rate = (total_runs / balls_faced * 100) if balls_faced > 0 else 0
    
    # Base points
    points = total_runs * 1  # Run points
    points += boundaries * 1  # Boundary bonus
    points += sixes * 2      # Six bonus
    
    # Milestone bonuses (only if not a century)
    if total_runs >= 100:
        points += 16  # Century bonus only
    elif total_runs >= 50:
        points += 8   # Half-century bonus
    elif total_runs >= 30:
        points += 4   # 30-run bonus
    
    # Duck penalty
    if total_runs == 0 and balls_faced > 0:
        points -= 2
    
    # Strike rate points (min 10 balls)
    if balls_faced >= 10:
        if strike_rate > 170:
            points += 6
        elif strike_rate > 150:
            points += 4
        elif strike_rate > 130:
            points += 2
        elif strike_rate < 50:
            points -= 6
        elif strike_rate < 60:
            points -= 4
        elif strike_rate < 70:
            points -= 2
    
    return {
        'total_points': points,
        'runs': total_runs,
        'boundaries': boundaries,
        'sixes': sixes,
        'strike_rate': strike_rate,
        'balls_faced': balls_faced
    }

def calculate_bowling_points(player_deliveries: pd.DataFrame) -> Dict[str, float]:
    """Calculate bowling points for a player in a match"""
    if player_deliveries.empty:
        return {
            'total_points': 0.0,
            'wickets': 0,
            'lbw_bowled': 0,
            'maidens': 0,
            'economy_rate': 0.0,
            'overs_bowled': 0
        }
    
    # Calculate basic stats
    wickets = len(player_deliveries[
        (player_deliveries['is_wicket'] == 1) & 
        (player_deliveries['wicket_kind'] != 'run out')
    ])
    
    lbw_bowled = len(player_deliveries[
        player_deliveries['wicket_kind'].isin(['lbw', 'bowled'])
    ])
    
    # Calculate maidens
    overs = player_deliveries.groupby('over')
    maidens = sum(1 for _, over in overs if over['total_runs'].sum() == 0)
    
    # Calculate economy rate
    total_overs = len(player_deliveries) / 6
    total_runs = player_deliveries['total_runs'].sum()
    economy_rate = total_runs / total_overs if total_overs > 0 else 0
    
    # Base points
    points = wickets * 25  # Wicket points
    points += lbw_bowled * 8  # LBW/Bowled bonus
    points += maidens * 12  # Maiden over points
    
    # Wicket milestone bonuses
    if wickets >= 5:
        points += 16
    elif wickets >= 4:
        points += 8
    elif wickets >= 3:
        points += 4
    
    # Economy rate points (min 2 overs)
    if total_overs >= 2:
        if economy_rate < 5:
            points += 6
        elif economy_rate < 6:
            points += 4
        elif economy_rate < 7:
            points += 2
        elif economy_rate >= 10 and economy_rate < 11:
            points -= 2
        elif economy_rate >= 11 and economy_rate < 12:
            points -= 4
        elif economy_rate >= 12:
            points -= 6
    
    return {
        'total_points': points,
        'wickets': wickets,
        'lbw_bowled': lbw_bowled,
        'maidens': maidens,
        'economy_rate': economy_rate,
        'overs_bowled': total_overs
    }

def calculate_fielding_points(player_fielding: pd.DataFrame) -> Dict[str, float]:
    """Calculate fielding points for a player in a match"""
    if player_fielding.empty:
        return {
            'total_points': 0.0,
            'catches': 0,
            'stumpings': 0,
            'run_outs': 0
        }
    
    # Calculate basic stats
    catches = len(player_fielding[player_fielding['wicket_kind'] == 'caught'])
    stumpings = len(player_fielding[player_fielding['wicket_kind'] == 'stumped'])
    run_outs = len(player_fielding[player_fielding['wicket_kind'] == 'run out'])
    
    # Base points
    points = catches * 8  # Catch points
    points += stumpings * 12  # Stumping points
    points += run_outs * 6  # Run out points (assuming all are not direct hits)
    
    # 3 catch bonus
    if catches >= 3:
        points += 4
    
    return {
        'total_points': points,
        'catches': catches,
        'stumpings': stumpings,
        'run_outs': run_outs
    }

def process_match_data(match_id: int, deliveries_df: pd.DataFrame) -> List[Dict]:
    """Process a single match and calculate points for all players"""
    match_deliveries = deliveries_df[deliveries_df['match_id'] == match_id]
    
    # Get all players in the match
    all_players = pd.concat([
        match_deliveries['batter'],
        match_deliveries['bowler'],
        match_deliveries['non_striker']
    ]).unique()
    
    player_stats = []
    
    for player in all_players:
        # Batting stats
        batting_deliveries = match_deliveries[match_deliveries['batter'] == player]
        batting_stats = calculate_batting_points(batting_deliveries)
        
        # Bowling stats
        bowling_deliveries = match_deliveries[match_deliveries['bowler'] == player]
        bowling_stats = calculate_bowling_points(bowling_deliveries)
        
        # Fielding stats
        fielding_deliveries = match_deliveries[match_deliveries['fielder'] == player]
        fielding_stats = calculate_fielding_points(fielding_deliveries)
        
        # Calculate total points
        total_points = (
            batting_stats['total_points'] +
            bowling_stats['total_points'] +
            fielding_stats['total_points']
        )
        
        # Determine player role based on point distribution
        batting_ratio = batting_stats['total_points'] / total_points if total_points > 0 else 0
        bowling_ratio = bowling_stats['total_points'] / total_points if total_points > 0 else 0
        
        if batting_ratio > 0.7:
            role = "Batsman"
        elif bowling_ratio > 0.7:
            role = "Bowler"
        elif batting_ratio > 0.3 and bowling_ratio > 0.3:
            role = "All-Rounder"
        elif batting_ratio > bowling_ratio:
            role = "Batting All-Rounder"
        else:
            role = "Bowling All-Rounder"
        
        player_stats.append({
            'match_id': match_id,
            'player': player,
            'role': role,
            'total_points': total_points,
            'batting_points': batting_stats['total_points'],
            'bowling_points': bowling_stats['total_points'],
            'fielding_points': fielding_stats['total_points'],
            # Batting details
            'runs': batting_stats['runs'],
            'boundaries': batting_stats['boundaries'],
            'sixes': batting_stats['sixes'],
            'strike_rate': batting_stats['strike_rate'],
            'balls_faced': batting_stats['balls_faced'],
            # Bowling details
            'wickets': bowling_stats['wickets'],
            'lbw_bowled': bowling_stats['lbw_bowled'],
            'maidens': bowling_stats['maidens'],
            'economy_rate': bowling_stats['economy_rate'],
            'overs_bowled': bowling_stats['overs_bowled'],
            # Fielding details
            'catches': fielding_stats['catches'],
            'stumpings': fielding_stats['stumpings'],
            'run_outs': fielding_stats['run_outs']
        })
    
    return player_stats

def clean_dream_team_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and format dream team stats before saving."""
    # Remove any newlines from column names
    df.columns = [col.strip() for col in df.columns]
    
    # Fill any missing values
    df = df.fillna(0)
    
    # Ensure all numeric columns are properly formatted
    numeric_columns = ['total_points', 'batting_points', 'bowling_points', 'fielding_points',
                      'runs', 'boundaries', 'sixes', 'strike_rate', 'balls_faced',
                      'wickets', 'lbw_bowled', 'maidens', 'economy_rate', 'overs_bowled',
                      'catches', 'stumpings', 'run_outs']
    
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    return df

def create_dream_team_dataset():
    """Create and save the dream team dataset."""
    # Get the current directory
    current_dir = Path(__file__).resolve().parent
    # Navigate to the app/data directory
    app_data_dir = current_dir.parent / 'app' / 'data'
    os.makedirs(app_data_dir, exist_ok=True)

    # Update paths to read from and write to app/data directory
    matches_df = pd.read_parquet(app_data_dir / 'matches.parquet')
    deliveries_df = pd.read_parquet(app_data_dir / 'deliveries.parquet')
    
    return create_dream_team_dataset_from_dataframes(matches_df, deliveries_df)

def create_dream_team_dataset_from_dataframes(matches_df, deliveries_df):
    """Create and save the dream team dataset from provided dataframes."""
    # Get the current directory
    current_dir = Path(__file__).resolve().parent
    # Navigate to the app/data directory
    app_data_dir = current_dir.parent / 'app' / 'data'
    os.makedirs(app_data_dir, exist_ok=True)
    
    all_match_stats = []
    for match_id in matches_df['match_id'].unique():
        match_deliveries = deliveries_df[deliveries_df['match_id'] == match_id]
        match_stats = process_match_data(match_id, match_deliveries)
        all_match_stats.extend(match_stats)
    
    dream_team_df = pd.DataFrame(all_match_stats)
    
    # Clean and format the stats
    dream_team_df = clean_dream_team_stats(dream_team_df)
    
    # Save to Parquet with compression
    dream_team_df.to_parquet(app_data_dir / 'dream_team_stats.parquet', compression='snappy', index=False)
    return dream_team_df

def compute_dream_team():
    """Compute dream team data and return the data frame."""
    return create_dream_team_dataset()

if __name__ == "__main__":
    create_dream_team_dataset() 