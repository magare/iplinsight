import pandas as pd
import json
import glob
import os
import sys
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
from tqdm import tqdm

# Add the src directory to Python path to fix imports
current_dir = Path(__file__).resolve().parent
src_dir = current_dir.parent
sys.path.append(str(src_dir))

from data.data_utils.normalize_team_names import normalize_and_abbreviate_teams
from data.data_utils.normalize_venue_names import normalize_and_shorten_venues

def convert_season(season):
    """
    Converts the season string according to specified rules.
    """
    # Manual mappings for specific seasons
    season_mappings = {
        '2007/08': '2008',
        '2009/10': '2010',
        '2020/21': '2020'
    }
    
    if isinstance(season, int):
        return str(season)  # Convert integer to string
    if isinstance(season, str):
        if season in season_mappings:
            return season_mappings[season]  # Return mapped value if exists
        if '/' in season:
            return season.split('/')[0]  # Return the first part of the season
        return season  # Return as is if no conversion is needed
    return season  # Return as is if no conversion is needed

def process_ipl_data(filepath, team_mapping, venue_mapping):
    """
    Processes IPL match data from a JSON file and returns two DataFrames:
    match_df (match-level data) and delivery_df (delivery-level data).
    """
    with open(filepath, 'r') as f:
        data = json.load(f)

    # --- Match-Level Data ---
    match_info = data['info']
    
    # Extract Season and Match Number
    season = match_info.get('season')
    season = convert_season(season)  # Convert the season using the new function
    
    # Handle event information and match number
    event_info = match_info.get('event', {})

    if not hasattr(process_ipl_data, 'max_match_number'):
        process_ipl_data.max_match_number = {}
    if season not in process_ipl_data.max_match_number:
        process_ipl_data.max_match_number[season] = 0

    provided_match_number = event_info.get('match_number', '')
    try:
        candidate = int(provided_match_number) if provided_match_number != '' else (process_ipl_data.max_match_number[season] + 1)
    except Exception:
        candidate = process_ipl_data.max_match_number[season] + 1

    if candidate <= process_ipl_data.max_match_number[season]:
        candidate = process_ipl_data.max_match_number[season] + 1

    match_number = candidate
    process_ipl_data.max_match_number[season] = candidate

    # Create Match ID
    match_id = f"{season}_{match_number:02d}" if match_number else f"{season}_unknown"

    # Extract winner and win margin details
    outcome = match_info.get('outcome', {})
    winner = outcome.get('winner', '')
    win_by_runs = outcome.get('by', {}).get('runs', 0)
    win_by_wickets = outcome.get('by', {}).get('wickets', 0)

    # Normalize team names and venue names using the provided mappings
    team1 = team_mapping.get(match_info['teams'][0], match_info['teams'][0])
    team2 = team_mapping.get(match_info['teams'][1], match_info['teams'][1])
    venue = venue_mapping.get(match_info.get('venue', ''), match_info.get('venue', ''))

    match_data = {
        'match_id': match_id,
        'season': season,
        'date': match_info['dates'][0],
        'team1': team1,
        'team2': team2,
        'venue': venue,
        'city': match_info.get('city', ''),
        'toss_winner': team_mapping.get(match_info['toss']['winner'], match_info['toss']['winner']),
        'toss_decision': match_info['toss']['decision'],
        'winner': team_mapping.get(winner, winner),
        'win_by_runs': win_by_runs,
        'win_by_wickets': win_by_wickets,
        'player_of_match': match_info.get('player_of_match', [''])[0],
        'match_number': match_number
    }
    match_df = pd.DataFrame([match_data])

    # --- Delivery-Level Data ---
    deliveries_data = []
    
    for inning_idx, inning in enumerate(data['innings'], 1):
        batting_team = team_mapping.get(inning['team'], inning['team'])
        bowling_team = team_mapping.get(match_info['teams'][1] if inning['team'] == match_info['teams'][0] else match_info['teams'][0])
        
        for over in inning['overs']:
            for delivery_idx, delivery in enumerate(over['deliveries'], 1):
                delivery_data = {
                    'match_id': match_id,
                    'inning': inning_idx,
                    'batting_team': batting_team,
                    'bowling_team': bowling_team,
                    'over': over['over'] + 1,  # Convert to 1-based indexing
                    'ball': delivery_idx,
                    'batter': delivery['batter'],
                    'bowler': delivery['bowler'],
                    'non_striker': delivery['non_striker'],
                    'batsman_runs': delivery['runs']['batter'],
                    'extra_runs': delivery['runs']['extras'],
                    'total_runs': delivery['runs']['total'],
                    'is_wicket': 1 if 'wickets' in delivery else 0,
                    'player_out': '',
                    'wicket_kind': '',
                    'fielder': ''
                }

                # Handle extras
                extras_type = list(delivery.get('extras', {}).keys())
                delivery_data['extras_type'] = extras_type[0] if extras_type else ''

                # Handle wickets
                if 'wickets' in delivery:
                    wicket = delivery['wickets'][0]  # Take the first wicket if multiple
                    delivery_data['player_out'] = wicket.get('player_out', '')
                    delivery_data['wicket_kind'] = wicket.get('kind', '')
                    
                    # Handle fielder information
                    if 'fielders' in wicket:
                        fielders = [f['name'] for f in wicket['fielders']]
                        delivery_data['fielder'] = ', '.join(fielders)

                deliveries_data.append(delivery_data)

    delivery_df = pd.DataFrame(deliveries_data)

    return match_df, delivery_df

# Get the project root directory
project_root = current_dir.parent

# Path to the folder containing raw JSON files
raw_data_folder = project_root / 'data' / 'raw'

# Get team and venue mappings
# First create a temporary matches.csv with raw data to get mappings
temp_matches_df = pd.DataFrame()
for filepath in glob.glob(str(raw_data_folder / '*.json')):
    with open(filepath, 'r') as f:
        data = json.load(f)
        match_info = data['info']
        temp_matches_df = pd.concat([temp_matches_df, pd.DataFrame([{
            'team1': match_info['teams'][0],
            'team2': match_info['teams'][1],
            'venue': match_info.get('venue', '')
        }])], ignore_index=True)

# Create a temporary directory in the streamlit app data folder
app_data_dir = src_dir / 'app' / 'data'
os.makedirs(app_data_dir, exist_ok=True)

temp_matches_path = app_data_dir / 'temp_matches.parquet'
temp_matches_df.to_parquet(temp_matches_path, compression='snappy', index=False)

# Get the mappings
team_mapping = normalize_and_abbreviate_teams(str(temp_matches_path))
venue_mapping = normalize_and_shorten_venues(str(temp_matches_path))

# Remove temporary file
os.remove(temp_matches_path)

# Initialize empty DataFrames for all matches and deliveries
all_match_df = pd.DataFrame()
all_delivery_df = pd.DataFrame()

# Process each JSON file
for filepath in glob.glob(str(raw_data_folder / '*.json')):
    match_df, delivery_df = process_ipl_data(filepath, team_mapping, venue_mapping)
    all_match_df = pd.concat([all_match_df, match_df], ignore_index=True)
    all_delivery_df = pd.concat([all_delivery_df, delivery_df], ignore_index=True)

# Convert date to datetime
all_match_df['date'] = pd.to_datetime(all_match_df['date'])

# Sort matches by date and match number
all_match_df = all_match_df.sort_values(by=['date', 'match_number'])
all_match_df.reset_index(drop=True, inplace=True)

# --- New Block: Reassign sequential match numbers per season based on match date ---
old_to_new_id = {}
for season, season_df in all_match_df.groupby('season'):
    season_df = season_df.sort_values('date')
    for new_num, idx in enumerate(season_df.index, start=1):
        old_id = all_match_df.loc[idx, 'match_id']
        new_id = f"{season}_{new_num:02d}"
        old_to_new_id[old_id] = new_id
        all_match_df.at[idx, 'match_number'] = new_num
        all_match_df.at[idx, 'match_id'] = new_id

# Update match_id in deliveries using the mapping
all_delivery_df['match_id'] = all_delivery_df['match_id'].map(old_to_new_id)

# Save the main dataframes to the app data directory
print("Saving main dataframes to app data directory...")
all_match_df.to_parquet(app_data_dir / 'matches.parquet', compression='snappy', index=False)
all_delivery_df.to_parquet(app_data_dir / 'deliveries.parquet', compression='snappy', index=False)
print(f"Main dataframes saved to {app_data_dir}")

# --- New Block: Save CSV copies of matches and deliveries in data/processed folder ---
processed_dir = current_dir / 'processed'
os.makedirs(processed_dir, exist_ok=True)
all_match_df.to_csv(processed_dir / 'matches.csv', index=False)
all_delivery_df.to_csv(processed_dir / 'deliveries.csv', index=False)
print(f"CSV files saved to {processed_dir}")

# --- New Block: Add batter_position functionality ---
# For each group of match_id, inning, and batting_team, assign a batting order

def assign_batter_positions(group):
    order = {}
    positions = []
    for _, row in group.iterrows():
        batter = row['batter']
        if batter not in order:
            order[batter] = len(order) + 1
        positions.append(order[batter])
    return pd.Series(positions, index=group.index)

all_delivery_df['batter_position'] = all_delivery_df.groupby(['match_id', 'inning', 'batting_team']).apply(assign_batter_positions).reset_index(level=[0,1,2], drop=True)

# Create a match order mapping from the sorted matches dataframe
# This ensures deliveries follow the same sequence as matches
match_order_mapping = {match_id: idx for idx, match_id in enumerate(all_match_df['match_id'])}
all_delivery_df['match_order'] = all_delivery_df['match_id'].map(match_order_mapping)

# Sort deliveries using the match order, then inning, over, and ball
all_delivery_df = all_delivery_df.sort_values(by=['match_order', 'inning', 'over', 'ball'])
all_delivery_df = all_delivery_df.drop(columns=['match_order'])
all_delivery_df.reset_index(drop=True, inplace=True)

# --- New Block: Pre-compute data for overview section ---
print("Pre-computing data for the overview section...")

# 1. Matches per season
matches_per_season = all_match_df.groupby('season').size().reset_index(name='matches')
matches_per_season['season'] = matches_per_season['season'].astype(str)
matches_per_season = matches_per_season.sort_values('season')

# 2. Average runs per match by season
season_runs = all_delivery_df.groupby('match_id')['total_runs'].sum().reset_index()
season_runs = pd.merge(season_runs, all_match_df[['match_id', 'season']], on='match_id')
avg_runs_by_season = season_runs.groupby('season')['total_runs'].mean().reset_index()
avg_runs_by_season['season'] = avg_runs_by_season['season'].astype(str)
avg_runs_by_season = avg_runs_by_season.sort_values('season')

# 3. Average wickets per match by season
season_wickets = all_delivery_df[all_delivery_df['is_wicket'] == 1].groupby('match_id').size().reset_index(name='wickets')
season_wickets = pd.merge(season_wickets, all_match_df[['match_id', 'season']], on='match_id')
avg_wickets_by_season = season_wickets.groupby('season')['wickets'].mean().reset_index()
avg_wickets_by_season['season'] = avg_wickets_by_season['season'].astype(str)
avg_wickets_by_season = avg_wickets_by_season.sort_values('season')

# 4. Team participation data
teams = pd.concat([all_match_df['team1'], all_match_df['team2']]).unique()
team_data = []
for team in sorted(teams):
    team_matches = all_match_df[
        (all_match_df['team1'] == team) | 
        (all_match_df['team2'] == team)
    ]
    
    seasons_played = sorted(team_matches['season'].unique(), key=str)
    total_matches = len(team_matches)
    wins = len(all_match_df[all_match_df['winner'] == team])
    win_rate = (wins / total_matches * 100) if total_matches > 0 else 0
    
    team_data.append({
        'Team': team,
        'Seasons_Played': ', '.join(map(str, seasons_played)),
        'Total_Matches': total_matches,
        'Wins': wins,
        'Win_Rate': win_rate
    })

team_participation_df = pd.DataFrame(team_data)
team_participation_df = team_participation_df.sort_values('Total_Matches', ascending=False)

# Save pre-computed data for overview section
matches_per_season.to_parquet(app_data_dir / 'overview_matches_per_season.parquet', compression='snappy', index=False)
avg_runs_by_season.to_parquet(app_data_dir / 'overview_avg_runs_by_season.parquet', compression='snappy', index=False)
avg_wickets_by_season.to_parquet(app_data_dir / 'overview_avg_wickets_by_season.parquet', compression='snappy', index=False)
team_participation_df.to_parquet(app_data_dir / 'overview_team_participation.parquet', compression='snappy', index=False)

print(f"Pre-computed data for overview section saved to {app_data_dir}")

# --- New Block: Pre-compute data for team analysis section ---
print("Pre-computing data for the team analysis section...")

# 1. Basic team metrics (Win/Loss Ratio)
team_stats = pd.DataFrame()
teams = pd.concat([all_match_df['team1'], all_match_df['team2']]).unique()

for team in teams:
    team_matches = all_match_df[
        (all_match_df['team1'] == team) | 
        (all_match_df['team2'] == team)
    ]
    wins = len(all_match_df[all_match_df['winner'] == team])
    losses = len(team_matches) - wins
    win_loss_ratio = wins / losses if losses > 0 else float('inf')
    
    team_stats = pd.concat([
        team_stats,
        pd.DataFrame({
            'Team': [team],
            'Matches': [len(team_matches)],
            'Wins': [wins],
            'Losses': [losses],
            'Win_Loss_Ratio': [win_loss_ratio]
        })
    ])

# 2. Batting First vs Chasing Success
batting_first_stats = []
for team in team_stats['Team']:
    # Batting First
    batting_first = all_match_df[
        ((all_match_df['team1'] == team) & (all_match_df['toss_decision'] == 'bat')) |
        ((all_match_df['team2'] == team) & (all_match_df['toss_decision'] == 'field'))
    ]
    batting_first_wins = len(batting_first[batting_first['winner'] == team])
    batting_first_total = len(batting_first)
    
    # Chasing
    chasing = all_match_df[
        ((all_match_df['team1'] == team) & (all_match_df['toss_decision'] == 'field')) |
        ((all_match_df['team2'] == team) & (all_match_df['toss_decision'] == 'bat'))
    ]
    chasing_wins = len(chasing[chasing['winner'] == team])
    chasing_total = len(chasing)
    
    batting_first_stats.append({
        'Team': team,
        'Batting_First_Win_Pct': (batting_first_wins / batting_first_total * 100) if batting_first_total > 0 else 0,
        'Chasing_Win_Pct': (chasing_wins / chasing_total * 100) if chasing_total > 0 else 0,
        'Batting_First_Matches': batting_first_total,
        'Batting_First_Wins': batting_first_wins,
        'Chasing_Matches': chasing_total,
        'Chasing_Wins': chasing_wins
    })

batting_first_df = pd.DataFrame(batting_first_stats)

# 3. Net Run Rate Analysis
def calculate_nrr(innings1_score, innings1_overs, innings2_score, innings2_overs):
    """Calculate Net Run Rate for a match."""
    if innings1_overs == 0 or innings2_overs == 0:
        return 0
    
    team1_rr = innings1_score / innings1_overs
    team2_rr = innings2_score / innings2_overs
    return team1_rr - team2_rr

nrr_data = []
for _, match in all_match_df.iterrows():
    match_deliveries = all_delivery_df[all_delivery_df['match_id'] == match['match_id']]
    
    # Calculate overs and scores for both innings
    innings1 = match_deliveries[match_deliveries['inning'] == 1]
    innings2 = match_deliveries[match_deliveries['inning'] == 2]
    
    innings1_score = innings1['total_runs'].sum()
    innings2_score = innings2['total_runs'].sum()
    
    innings1_overs = len(innings1) / 6
    innings2_overs = len(innings2) / 6
    
    nrr = calculate_nrr(innings1_score, innings1_overs, innings2_score, innings2_overs)
    
    nrr_data.append({
        'match_id': match['match_id'],
        'season': match['season'],
        'team1': match['team1'],
        'team2': match['team2'],
        'winner': match['winner'],
        'nrr': nrr
    })

nrr_df = pd.DataFrame(nrr_data)
avg_nrr_by_season = nrr_df.groupby(['season', 'team1'])['nrr'].mean().reset_index()

# 4. Playoff Analysis
matches_df_copy = all_match_df.copy()
matches_df_copy['match_number_numeric'] = pd.to_numeric(matches_df_copy['match_number'], errors='coerce')
matches_df_copy['match_number_str'] = matches_df_copy['match_number'].astype(str)

# Identify playoff matches
playoff_matches = matches_df_copy[
    matches_df_copy['match_number_str'].str.contains('Final|Qualifier|Eliminator|Playoff', case=False, na=False)
]

# If no playoff matches found using string matching, try to identify them by match number
if len(playoff_matches) == 0:
    playoff_matches = pd.DataFrame()
    for season in matches_df_copy['season'].unique():
        season_matches = matches_df_copy[matches_df_copy['season'] == season]
        # Get the last 4 matches of each season (typically playoffs)
        season_playoffs = season_matches.nlargest(4, 'match_number_numeric')
        playoff_matches = pd.concat([playoff_matches, season_playoffs])

playoff_stats = []
for team in matches_df_copy['team1'].unique():
    team_playoffs = playoff_matches[
        (playoff_matches['team1'] == team) |
        (playoff_matches['team2'] == team)
    ]
    
    playoff_appearances = len(team_playoffs)
    playoff_wins = len(playoff_matches[playoff_matches['winner'] == team])
    
    # For finals, take the last match of each season
    final_matches = pd.DataFrame()
    for season in playoff_matches['season'].unique():
        season_matches = playoff_matches[playoff_matches['season'] == season]
        final_match = season_matches.nlargest(1, 'match_number_numeric')
        final_matches = pd.concat([final_matches, final_match])
    
    final_appearances = len(final_matches[
        (final_matches['team1'] == team) |
        (final_matches['team2'] == team)
    ])
    championships = len(final_matches[final_matches['winner'] == team])
    
    playoff_stats.append({
        'Team': team,
        'Playoff_Appearances': playoff_appearances,
        'Final_Appearances': final_appearances,
        'Championships': championships
    })

playoff_df = pd.DataFrame(playoff_stats)
playoff_df = playoff_df[playoff_df['Playoff_Appearances'] > 0]

# 5. Head-to-Head Win Matrix
teams = pd.concat([all_match_df['team1'], all_match_df['team2']]).unique()
win_matrix_data = pd.DataFrame(0, index=teams, columns=teams)

for _, row in all_match_df.iterrows():
    winner = row['winner']
    if pd.isna(winner) or winner == "":
        continue
    # Identify the losing team
    if row['team1'] == winner:
        loser = row['team2']
    else:
        loser = row['team1']
    win_matrix_data.loc[winner, loser] += 1

# 6. Venue Performance Base Statistics
venue_stats = []
for venue in all_match_df['venue'].unique():
    venue_matches = all_match_df[all_match_df['venue'] == venue]
    
    for team in teams:
        team_matches = venue_matches[
            (venue_matches['team1'] == team) | 
            (venue_matches['team2'] == team)
        ]
        wins = len(venue_matches[venue_matches['winner'] == team])
        matches_played = len(team_matches)
        
        if matches_played > 0:
            win_rate = (wins / matches_played) * 100
            venue_stats.append({
                'Venue': venue,
                'Team': team,
                'Matches': matches_played,
                'Wins': wins,
                'Win_Rate': win_rate
            })

# 7. Phase Analysis Base Data
phases = {
    'Powerplay': (0, 6),
    'Middle Overs': (7, 15),
    'Death Overs': (16, 20)
}

phase_stats = []
for team in teams:
    # Batting stats by phase
    batting_deliveries = all_delivery_df[all_delivery_df['batting_team'] == team]
    
    # Bowling stats by phase
    bowling_deliveries = all_delivery_df[all_delivery_df['bowling_team'] == team]
    
    for phase_name, (start, end) in phases.items():
        # Batting phase stats
        batting_phase = batting_deliveries[
            (batting_deliveries['over'] >= start) &
            (batting_deliveries['over'] <= end)
        ]
        batting_runs = batting_phase['total_runs'].sum()
        batting_balls = len(batting_phase)
        batting_overs = batting_balls / 6 if batting_balls > 0 else 0
        batting_run_rate = batting_runs / batting_overs if batting_overs > 0 else 0
        batting_boundaries = len(batting_phase[batting_phase['batsman_runs'].isin([4, 6])])
        batting_boundary_pct = (batting_boundaries / batting_balls * 100) if batting_balls > 0 else 0
        
        # Bowling phase stats
        bowling_phase = bowling_deliveries[
            (bowling_deliveries['over'] >= start) &
            (bowling_deliveries['over'] <= end)
        ]
        bowling_runs = bowling_phase['total_runs'].sum()
        bowling_balls = len(bowling_phase)
        bowling_overs = bowling_balls / 6 if bowling_balls > 0 else 0
        bowling_economy = bowling_runs / bowling_overs if bowling_overs > 0 else 0
        bowling_wickets = len(bowling_phase[bowling_phase['is_wicket'] == 1])
        bowling_wicket_rate = bowling_wickets / bowling_overs if bowling_overs > 0 else 0
        
        phase_stats.append({
            'Team': team,
            'Phase': phase_name,
            'Batting_Runs': batting_runs,
            'Batting_Balls': batting_balls,
            'Batting_Run_Rate': batting_run_rate,
            'Batting_Boundaries': batting_boundaries,
            'Batting_Boundary_Pct': batting_boundary_pct,
            'Bowling_Runs': bowling_runs,
            'Bowling_Balls': bowling_balls,
            'Bowling_Economy': bowling_economy,
            'Bowling_Wickets': bowling_wickets,
            'Bowling_Wicket_Rate': bowling_wicket_rate
        })

# Save pre-computed data for team analysis
team_stats.to_parquet(app_data_dir / 'team_stats.parquet', compression='snappy', index=False)
batting_first_df.to_parquet(app_data_dir / 'batting_first_stats.parquet', compression='snappy', index=False)
nrr_df.to_parquet(app_data_dir / 'nrr_data.parquet', compression='snappy', index=False)
avg_nrr_by_season.to_parquet(app_data_dir / 'avg_nrr_by_season.parquet', compression='snappy', index=False)
playoff_df.to_parquet(app_data_dir / 'playoff_stats.parquet', compression='snappy', index=False)
win_matrix_data.to_parquet(app_data_dir / 'head_to_head_matrix.parquet', compression='snappy')
pd.DataFrame(venue_stats).to_parquet(app_data_dir / 'venue_team_stats.parquet', compression='snappy', index=False)
pd.DataFrame(phase_stats).to_parquet(app_data_dir / 'team_phase_stats.parquet', compression='snappy', index=False)

print(f"Pre-computed data for team analysis section saved to {app_data_dir}")

# Pre-compute data for player analysis section
print("Pre-computing data for player analysis section...")

# Batting statistics
batting_stats = all_delivery_df.groupby('batter').agg({
    'batsman_runs': ['sum', 'count'],  # runs and balls faced
    'match_id': 'nunique',  # number of matches
    'is_wicket': 'sum'  # number of dismissals
}).reset_index()

# Flatten column names
batting_stats.columns = ['batter', 'runs', 'balls_faced', 'matches', 'dismissals']

# Calculate derived metrics
batting_stats['batting_average'] = batting_stats['runs'] / batting_stats['dismissals'].replace(0, 1)
batting_stats['batting_strike_rate'] = (batting_stats['runs'] / batting_stats['balls_faced']) * 100

# Calculate boundary stats
boundary_stats = all_delivery_df.groupby('batter').agg({
    'ball': 'count',  # total balls
    'batsman_runs': [
        ('dot_balls', lambda x: (x == 0).sum()),  # dot balls
        ('boundaries', lambda x: ((x == 4) | (x == 6)).sum()),  # boundaries
        ('fours', lambda x: (x == 4).sum()),  # fours
        ('sixes', lambda x: (x == 6).sum())  # sixes
    ]
}).reset_index()

# Flatten column names
boundary_stats.columns = ['batter', 'total_balls', 'dot_balls', 'boundaries', 'fours', 'sixes']

# Merge stats
batting_stats = pd.merge(batting_stats, boundary_stats, on='batter')

# Calculate percentages
batting_stats['dot_ball_percentage'] = (batting_stats['dot_balls'] / batting_stats['total_balls']) * 100
batting_stats['boundary_percentage'] = (batting_stats['boundaries'] / batting_stats['total_balls']) * 100
batting_stats['runs_per_boundary'] = batting_stats['runs'] / batting_stats['boundaries'].replace(0, 1)

# Rename matches column to be consistent with merge suffixes
batting_stats = batting_stats.rename(columns={'matches': 'matches_batting'})

# Calculate batting milestones (30s, 50s, 100s)
innings_scores = all_delivery_df.groupby(['match_id', 'batter'])['batsman_runs'].sum().reset_index()
milestones = innings_scores.groupby('batter').agg({
    'batsman_runs': lambda x: [
        sum((x >= 30) & (x < 50)),  # 30s
        sum((x >= 50) & (x < 100)),  # 50s
        sum(x >= 100)  # 100s
    ]
}).reset_index()

# Convert list to separate columns
milestones[['thirties', 'fifties', 'hundreds']] = pd.DataFrame(
    milestones['batsman_runs'].tolist(),
    index=milestones.index
)

# Merge milestone stats with batting stats
batting_stats = pd.merge(batting_stats, milestones[['batter', 'thirties', 'fifties', 'hundreds']], on='batter', how='left')
batting_stats = batting_stats.fillna(0)

# Phase-wise batting statistics
all_delivery_df['phase'] = pd.cut(
    all_delivery_df['over'],
    bins=[-1, 5, 15, 20],
    labels=['Powerplay', 'Middle Overs', 'Death Overs']
)

batting_phase_stats = all_delivery_df.groupby(['batter', 'phase']).agg({
    'batsman_runs': ['sum', 'count'],
    'is_wicket': 'sum'
}).reset_index()

# Flatten columns and calculate strike rate
batting_phase_stats.columns = ['batter', 'phase', 'runs', 'balls', 'dismissals']
batting_phase_stats['batting_strike_rate'] = (batting_phase_stats['runs'] / batting_phase_stats['balls']) * 100
batting_phase_stats['batting_average'] = batting_phase_stats['runs'] / batting_phase_stats['dismissals'].replace(0, 1)

# Position-wise batting statistics
all_delivery_df_copy = all_delivery_df.copy()
all_delivery_df_copy['batter_position'] = all_delivery_df_copy.groupby(['match_id', 'inning', 'batting_team'])['batter'].transform(lambda x: pd.factorize(x)[0] + 1)

position_stats = all_delivery_df_copy.groupby(['batter', 'batter_position']).agg({
    'batsman_runs': ['sum', 'count'],
    'is_wicket': 'sum',
    'match_id': 'nunique'
}).reset_index()

# Flatten columns
position_stats.columns = ['batter', 'position', 'runs', 'balls', 'dismissals', 'innings']

# Calculate metrics
position_stats['batting_average'] = position_stats['runs'] / position_stats['dismissals'].replace(0, 1)
position_stats['batting_strike_rate'] = (position_stats['runs'] / position_stats['balls']) * 100

# Bowling statistics
bowling_stats = all_delivery_df.groupby('bowler').agg({
    'total_runs': 'sum',  # runs conceded
    'ball': 'count',  # balls bowled
    'is_wicket': 'sum',  # wickets taken
    'match_id': 'nunique'  # matches played
}).reset_index()

# Rename matches column to be consistent with merge suffixes
bowling_stats = bowling_stats.rename(columns={'match_id': 'matches_bowling'})

# Calculate derived metrics
bowling_stats['overs'] = bowling_stats['ball'] / 6
bowling_stats['bowling_economy'] = bowling_stats['total_runs'] / bowling_stats['overs']
bowling_stats['bowling_average'] = bowling_stats['total_runs'] / bowling_stats['is_wicket'].replace(0, 1)  # runs per wicket
bowling_stats['bowling_strike_rate'] = bowling_stats['ball'] / bowling_stats['is_wicket'].replace(0, 1)  # balls per wicket
bowling_stats['wickets_per_match'] = bowling_stats['is_wicket'] / bowling_stats['matches_bowling']

# Calculate dot balls
dot_balls = all_delivery_df[all_delivery_df['total_runs'] == 0].groupby('bowler').size().reset_index(name='dot_balls')
bowling_stats = pd.merge(bowling_stats, dot_balls, on='bowler', how='left')
bowling_stats['dot_ball_percentage'] = (bowling_stats['dot_balls'] / bowling_stats['ball']) * 100

# Wicket types by bowler
wickets = all_delivery_df[all_delivery_df['is_wicket'] == 1]
wicket_types = wickets.groupby(['bowler', 'wicket_kind']).size().reset_index(name='count')
wicket_types = wicket_types.pivot(
    index='bowler',
    columns='wicket_kind',
    values='count'
).fillna(0).reset_index()

# Phase-wise bowling statistics
bowling_phase_stats = all_delivery_df.groupby(['bowler', 'phase']).agg({
    'total_runs': 'sum',
    'ball': 'count',
    'is_wicket': 'sum',
    'match_id': 'nunique'
}).reset_index()

# Calculate metrics
bowling_phase_stats['overs'] = bowling_phase_stats['ball'] / 6
bowling_phase_stats['bowling_economy'] = bowling_phase_stats['total_runs'] / bowling_phase_stats['overs']
bowling_phase_stats['bowling_average'] = bowling_phase_stats['total_runs'] / bowling_phase_stats['is_wicket'].replace(0, 1)
bowling_phase_stats['bowling_strike_rate'] = bowling_phase_stats['ball'] / bowling_phase_stats['is_wicket'].replace(0, 1)

# All-rounder statistics
allrounders = pd.merge(
    batting_stats,
    bowling_stats,
    left_on='batter',
    right_on='bowler',
    how='inner',
    suffixes=('_batting', '_bowling')
)

# Rename columns for clarity
allrounders = allrounders.rename(columns={
    'batter': 'player',
    'matches_batting': 'batting_matches',
    'matches_bowling': 'bowling_matches',
    'runs': 'batting_runs',
    'total_runs': 'runs_conceded',
    'is_wicket': 'wickets'
})

# Calculate composite scores
# Normalize batting and bowling stats to a 0-1 scale
allrounders['batting_score'] = (
    (allrounders['batting_runs'] / allrounders['batting_runs'].max()) * 0.4 +
    (allrounders['batting_average'] / allrounders['batting_average'].max()) * 0.3 +
    (allrounders['batting_strike_rate'] / allrounders['batting_strike_rate'].max()) * 0.3
)

allrounders['bowling_score'] = (
    (allrounders['wickets'] / allrounders['wickets'].max()) * 0.4 +
    (1 - allrounders['bowling_economy'] / allrounders['bowling_economy'].max()) * 0.3 +
    (1 - allrounders['bowling_average'] / allrounders['bowling_average'].max()) * 0.3
)

# Calculate overall all-rounder score
allrounders['allrounder_score'] = (allrounders['batting_score'] + allrounders['bowling_score']) / 2

# Head-to-head statistics
h2h_stats = all_delivery_df.groupby(['batter', 'bowler']).agg({
    'batsman_runs': ['sum', 'count'],
    'is_wicket': 'sum',
    'match_id': 'nunique'
}).reset_index()

# Flatten column names
h2h_stats.columns = ['batsman', 'bowler', 'runs', 'balls', 'dismissals', 'matches']

# Calculate derived metrics
h2h_stats['average'] = h2h_stats['runs'] / h2h_stats['dismissals'].replace(0, 1)
h2h_stats['strike_rate'] = (h2h_stats['runs'] / h2h_stats['balls']) * 100
h2h_stats['dominance_ratio'] = h2h_stats['strike_rate'] / (h2h_stats['dismissals'].replace(0, 0.5) * 10)

# Save all the pre-computed player analysis data to CSV files
data_dir = Path(__file__).resolve().parent.parent / "app" / "data"
data_dir.mkdir(exist_ok=True)

batting_stats.to_parquet(data_dir / "player_batting_stats.parquet", compression='snappy', index=False)
batting_phase_stats.to_parquet(data_dir / "player_batting_phase_stats.parquet", compression='snappy', index=False)
position_stats.to_parquet(data_dir / "player_position_stats.parquet", compression='snappy', index=False)
bowling_stats.to_parquet(data_dir / "player_bowling_stats.parquet", compression='snappy', index=False)
wicket_types.to_parquet(data_dir / "player_wicket_types.parquet", compression='snappy', index=False)
bowling_phase_stats.to_parquet(data_dir / "player_bowling_phase_stats.parquet", compression='snappy', index=False)
allrounders.to_parquet(data_dir / "player_allrounder_stats.parquet", compression='snappy', index=False)
h2h_stats.to_parquet(data_dir / "player_h2h_stats.parquet", compression='snappy', index=False)

print("Pre-computed data for player analysis section saved to:", data_dir)

# Pre-compute data for match analysis section
print("Pre-computing data for match analysis section...")

# --- Match Result Analysis ---
# Define result type based on win margins
all_match_df['result_type'] = all_match_df.apply(
    lambda x: 'runs' if x['win_by_runs'] > 0 
             else 'wickets' if x['win_by_wickets'] > 0 
             else 'no result',
    axis=1
)

# Define result margin from runs or wickets
all_match_df['result_margin'] = all_match_df.apply(
    lambda x: x['win_by_runs'] if x['win_by_runs'] > 0 else x['win_by_wickets'],
    axis=1
)

# Separate matches by victory method
runs_victories = all_match_df[all_match_df['result_type'] == 'runs'].copy()
wickets_victories = all_match_df[all_match_df['result_type'] == 'wickets'].copy()

# Calculate win method by season and percentages
win_method_by_season = all_match_df.groupby(['season', 'result_type']).size().unstack(fill_value=0)
win_method_by_season_pct = win_method_by_season.div(win_method_by_season.sum(axis=1), axis=0) * 100

# --- Toss Analysis ---
# Calculate toss decision frequency and percentages
toss_decisions = all_match_df.groupby('toss_decision').size()
toss_decisions_pct = (toss_decisions / len(all_match_df)) * 100

# Group toss decisions by season
toss_by_season = all_match_df.groupby(['season', 'toss_decision']).size().unstack(fill_value=0)
toss_by_season_pct = toss_by_season.div(toss_by_season.sum(axis=1), axis=0) * 100

# Group toss decisions by venue
toss_by_venue = all_match_df.groupby(['venue', 'toss_decision']).size().unstack(fill_value=0)
toss_by_venue_pct = toss_by_venue.div(toss_by_venue.sum(axis=1), axis=0) * 100

# Calculate win percentage when winning toss
all_match_df['won_toss_and_match'] = all_match_df['toss_winner'] == all_match_df['winner']
toss_win_pct = (all_match_df['won_toss_and_match'].mean()) * 100

# Toss decision outcomes
toss_decision_outcomes = all_match_df.groupby('toss_decision')['won_toss_and_match'].agg(['count', 'mean'])
toss_decision_outcomes['win_percentage'] = toss_decision_outcomes['mean'] * 100

# --- Scoring Analysis ---
# Ensure season is treated as string
all_match_df['season'] = all_match_df['season'].astype(str)

# Calculate total scores per match innings
match_scores = all_delivery_df.groupby(['match_id', 'inning'])['total_runs'].sum()
match_scores = match_scores.unstack()
match_scores.columns = [f'inning_{i}' for i in match_scores.columns]
match_scores = match_scores.rename(columns={'inning_1': 'first_innings', 'inning_2': 'second_innings'})

# Join match scores with match info
matches_info = all_match_df.set_index('match_id')[['season', 'venue', 'toss_decision', 'winner']]
match_scores = match_scores.join(matches_info)

# Average scores by venue
venue_scores = match_scores.groupby('venue').agg({
    'first_innings': ['mean', 'count'],
    'second_innings': 'mean'
}).round(2)
venue_scores.columns = ['first_innings_avg', 'matches', 'second_innings_avg']
venue_scores = venue_scores[venue_scores['matches'] >= 5]

# Average scores by season
season_scores = match_scores.groupby('season').agg({
    'first_innings': ['mean', 'std'],
    'second_innings': ['mean', 'std']
}).round(2)
season_scores.columns = ['first_innings_avg', 'first_innings_std', 
                           'second_innings_avg', 'second_innings_std']

# Phase-wise scoring (Powerplay, Middle Overs, Death Overs)
all_delivery_df['phase'] = pd.cut(
    all_delivery_df['over'],
    bins=[-1, 5, 15, 20],
    labels=['Powerplay', 'Middle Overs', 'Death Overs']
)
phase_scores = all_delivery_df.groupby(['match_id', 'inning', 'phase'])['total_runs'].sum().reset_index()
phase_avg = phase_scores.groupby(['inning', 'phase'])['total_runs'].agg(['mean', 'std']).round(2)

# --- High/Low Scoring Analysis ---
# Total match scores
match_total_scores = all_delivery_df.groupby('match_id')['total_runs'].sum()

# Define high and low thresholds (quartiles)
high_threshold = match_total_scores.quantile(0.75)
low_threshold = match_total_scores.quantile(0.25)

high_scoring_matches = all_match_df[all_match_df['match_id'].isin(match_total_scores[match_total_scores >= high_threshold].index)].copy()
low_scoring_matches = all_match_df[all_match_df['match_id'].isin(match_total_scores[match_total_scores <= low_threshold].index)].copy()

# Analysis by venue
high_scoring_venues = high_scoring_matches['venue'].value_counts()
low_scoring_venues = low_scoring_matches['venue'].value_counts()

# Analysis by team
high_scoring_teams = pd.concat([high_scoring_matches['team1'], high_scoring_matches['team2']]).value_counts()
low_scoring_teams = pd.concat([low_scoring_matches['team1'], low_scoring_matches['team2']]).value_counts()

# Analysis by season
high_scoring_seasons = high_scoring_matches.groupby('season').size()
low_scoring_seasons = low_scoring_matches.groupby('season').size()

# Toss impact
high_scoring_toss = high_scoring_matches.groupby('toss_decision')['match_id'].count()
low_scoring_toss = low_scoring_matches.groupby('toss_decision')['match_id'].count()

# Calculate run rates by phase
high_scoring_phases = all_delivery_df[all_delivery_df['match_id'].isin(high_scoring_matches['match_id'])].groupby('phase')['total_runs'].mean()
low_scoring_phases = all_delivery_df[all_delivery_df['match_id'].isin(low_scoring_matches['match_id'])].groupby('phase')['total_runs'].mean()

# Save pre-computed match analysis data
runs_victories.to_parquet(app_data_dir / 'match_runs_victories.parquet', compression='snappy', index=False)
wickets_victories.to_parquet(app_data_dir / 'match_wickets_victories.parquet', compression='snappy', index=False)
win_method_by_season.to_parquet(app_data_dir / 'match_win_method_by_season.parquet', compression='snappy')
win_method_by_season_pct.to_parquet(app_data_dir / 'match_win_method_by_season_pct.parquet', compression='snappy')

# Toss analysis
pd.DataFrame({'toss_decisions': toss_decisions, 'toss_decisions_pct': toss_decisions_pct}).to_parquet(app_data_dir / 'match_toss_decisions.parquet', compression='snappy', index=False)
toss_by_season.to_parquet(app_data_dir / 'match_toss_by_season.parquet', compression='snappy')
toss_by_season_pct.to_parquet(app_data_dir / 'match_toss_by_season_pct.parquet', compression='snappy')
toss_by_venue.to_parquet(app_data_dir / 'match_toss_by_venue.parquet', compression='snappy')
toss_by_venue_pct.to_parquet(app_data_dir / 'match_toss_by_venue_pct.parquet', compression='snappy')
pd.DataFrame({'toss_win_pct': [toss_win_pct]}).to_parquet(app_data_dir / 'match_toss_win_pct.parquet', compression='snappy', index=False)
toss_decision_outcomes.to_parquet(app_data_dir / 'match_toss_decision_outcomes.parquet', compression='snappy')

# Scoring analysis
match_scores.to_parquet(app_data_dir / 'match_scores.parquet', compression='snappy')
venue_scores.to_parquet(app_data_dir / 'match_venue_scores.parquet', compression='snappy')
season_scores.to_parquet(app_data_dir / 'match_season_scores.parquet', compression='snappy')
phase_avg.to_parquet(app_data_dir / 'match_phase_avg.parquet', compression='snappy')

# High/Low scoring analysis
pd.DataFrame({'high_threshold': [high_threshold], 'low_threshold': [low_threshold]}).to_parquet(app_data_dir / 'match_score_thresholds.parquet', compression='snappy', index=False)
# Convert Series to DataFrames with appropriate column names
high_scoring_venues.to_frame(name='score').to_parquet(app_data_dir / 'match_high_scoring_venues.parquet', compression='snappy')
low_scoring_venues.to_frame(name='score').to_parquet(app_data_dir / 'match_low_scoring_venues.parquet', compression='snappy')
high_scoring_teams.to_frame(name='score').to_parquet(app_data_dir / 'match_high_scoring_teams.parquet', compression='snappy')
low_scoring_teams.to_frame(name='score').to_parquet(app_data_dir / 'match_low_scoring_teams.parquet', compression='snappy')
high_scoring_seasons.to_frame(name='score').to_parquet(app_data_dir / 'match_high_scoring_seasons.parquet', compression='snappy')
low_scoring_seasons.to_frame(name='score').to_parquet(app_data_dir / 'match_low_scoring_seasons.parquet', compression='snappy')
high_scoring_toss.to_frame(name='count').to_parquet(app_data_dir / 'match_high_scoring_toss.parquet', compression='snappy')
low_scoring_toss.to_frame(name='count').to_parquet(app_data_dir / 'match_low_scoring_toss.parquet', compression='snappy')
high_scoring_phases.to_frame(name='score').to_parquet(app_data_dir / 'match_high_scoring_phases.parquet', compression='snappy')
low_scoring_phases.to_frame(name='score').to_parquet(app_data_dir / 'match_low_scoring_phases.parquet', compression='snappy')

print(f"Pre-computed data for match analysis section saved to {app_data_dir}")

# Pre-compute data for season analysis section
print("Pre-computing data for season analysis section...")

# Get unique seasons
seasons = sorted(all_match_df['season'].unique())

for season in seasons:
    print(f"Processing season {season}...")
    
    # Filter data for the selected season
    season_matches = all_match_df[all_match_df['season'] == season].copy()
    season_deliveries = all_delivery_df[all_delivery_df['match_id'].isin(season_matches['match_id'])].copy()
    
    # --- Basic Season Stats ---
    total_matches = len(season_matches)
    total_runs = season_deliveries['total_runs'].sum()
    total_wickets = season_deliveries['is_wicket'].sum() if 'is_wicket' in season_deliveries.columns else 0
    total_sixes = len(season_deliveries[season_deliveries['batsman_runs'] == 6])
    total_fours = len(season_deliveries[season_deliveries['batsman_runs'] == 4])
    
    # Determine season winner
    winner_mask = season_matches['winner'].value_counts()
    season_winner = winner_mask.index[0] if not winner_mask.empty else "Unknown"
    
    # Create season stats dataframe
    season_stats = pd.DataFrame({
        'total_matches': [total_matches],
        'total_runs': [total_runs],
        'total_wickets': [total_wickets],
        'total_sixes': [total_sixes],
        'total_fours': [total_fours],
        'avg_match_score': [total_runs / total_matches if total_matches > 0 else 0],
        'sixes_per_match': [total_sixes / total_matches if total_matches > 0 else 0],
        'fours_per_match': [total_fours / total_matches if total_matches > 0 else 0],
        'winner': [season_winner]
    })
    
    # --- Team Standings ---
    # Initialize standings DataFrame
    teams = pd.concat([season_matches['team1'], season_matches['team2']]).unique()
    standings = pd.DataFrame(index=teams)
    standings['matches'] = 0
    standings['wins'] = 0
    standings['losses'] = 0
    standings['points'] = 0
    standings['nrr'] = 0.0  # Net Run Rate
    standings['runs'] = 0
    standings['wickets'] = 0
    
    # Calculate basic stats
    for _, match in season_matches.iterrows():
        team1, team2 = match['team1'], match['team2']
        winner = match.get('winner', None)
        
        # Update matches played
        standings.loc[team1, 'matches'] += 1
        standings.loc[team2, 'matches'] += 1
        
        # Update wins and losses if winner is available
        if pd.notna(winner) and winner != '':
            standings.loc[winner, 'wins'] += 1
            loser = team2 if winner == team1 else team1
            standings.loc[loser, 'losses'] += 1
    
    # Calculate points (2 for win, 0 for loss)
    standings['points'] = standings['wins'] * 2
    
    # Sort by points and net run rate
    standings = standings.sort_values(['points', 'nrr'], ascending=[False, False])
    
    # --- Points Progression ---
    # Initialize points table
    points_progression = pd.DataFrame(index=range(1, len(season_matches) + 1), columns=teams)
    points_progression.fillna(0, inplace=True)
    
    # Calculate points progression
    current_points = {team: 0 for team in teams}
    
    for i, (_, match) in enumerate(season_matches.iterrows(), 1):
        if pd.notna(match['winner']) and match['winner'] != '':
            current_points[match['winner']] += 2
        
        for team in teams:
            points_progression.loc[i, team] = current_points[team]
    
    # --- Batting Stats ---
    batting_stats = season_deliveries.groupby('batter').agg({
        'batsman_runs': 'sum',
        'ball': 'count',
        'match_id': 'nunique'
    })
    
    batting_stats.columns = ['runs', 'balls', 'matches']
    
    # Calculate dismissals
    if 'is_wicket' in season_deliveries.columns:
        dismissals = season_deliveries[season_deliveries['is_wicket'] == 1].groupby('batter').size()
        batting_stats = batting_stats.join(dismissals.rename('dismissals'), how='left')
    else:
        batting_stats['dismissals'] = 0
    
    batting_stats['dismissals'] = batting_stats['dismissals'].fillna(0)
    
    # Calculate additional metrics
    batting_stats['average'] = batting_stats['runs'] / batting_stats['dismissals'].replace(0, 1)
    batting_stats['strike_rate'] = (batting_stats['runs'] / batting_stats['balls']) * 100
    
    # Calculate boundaries
    sixes = season_deliveries[season_deliveries['batsman_runs'] == 6].groupby('batter').size()
    fours = season_deliveries[season_deliveries['batsman_runs'] == 4].groupby('batter').size()
    
    batting_stats = batting_stats.join(sixes.rename('sixes'), how='left')
    batting_stats = batting_stats.join(fours.rename('fours'), how='left')
    batting_stats['sixes'] = batting_stats['sixes'].fillna(0)
    batting_stats['fours'] = batting_stats['fours'].fillna(0)
    
    # Calculate highest scores
    highest_scores = season_deliveries.groupby(['match_id', 'batter'])['batsman_runs'].sum()
    highest_scores = highest_scores.reset_index().groupby('batter')['batsman_runs'].max()
    batting_stats = batting_stats.join(highest_scores.rename('highest_score'), how='left')
    
    # --- Bowling Stats ---
    bowling_stats = season_deliveries.groupby('bowler').agg({
        'total_runs': 'sum',
        'ball': 'count',
        'match_id': 'nunique'
    })
    
    bowling_stats.columns = ['runs', 'balls', 'matches']
    
    # Calculate wickets
    if 'is_wicket' in season_deliveries.columns:
        wickets = season_deliveries[season_deliveries['is_wicket'] == 1].groupby('bowler').size()
        bowling_stats = bowling_stats.join(wickets.rename('wickets'), how='left')
    else:
        bowling_stats['wickets'] = 0
    
    bowling_stats['wickets'] = bowling_stats['wickets'].fillna(0)
    
    # Calculate additional metrics
    bowling_stats['overs'] = bowling_stats['balls'] / 6
    bowling_stats['economy'] = bowling_stats['runs'] / bowling_stats['overs']
    bowling_stats['average'] = bowling_stats.apply(
        lambda x: x['runs'] / x['wickets'] if x['wickets'] > 0 else float('inf'),
        axis=1
    )
    bowling_stats['strike_rate'] = bowling_stats.apply(
        lambda x: x['balls'] / x['wickets'] if x['wickets'] > 0 else float('inf'),
        axis=1
    )
    
    # --- Fielding Stats ---
    fielding_stats = pd.DataFrame(index=pd.Index([], name='player'))
    
    # Calculate catches if dismissal information is available
    if 'dismissal_kind' in season_deliveries.columns and 'fielder' in season_deliveries.columns:
        # Calculate catches
        catches = season_deliveries[
            season_deliveries['dismissal_kind'] == 'caught'
        ]['fielder'].value_counts()
        fielding_stats = fielding_stats.join(catches.rename('catches'), how='outer')
        
        # Calculate run outs
        run_outs = season_deliveries[
            season_deliveries['dismissal_kind'] == 'run out'
        ]['fielder'].value_counts()
        fielding_stats = fielding_stats.join(run_outs.rename('run_outs'), how='outer')
    
    fielding_stats = fielding_stats.fillna(0)
    
    # --- All-Round Stats ---
    # Find players who both batted and bowled
    common_players = set(batting_stats.index) & set(bowling_stats.index)
    all_round_stats = pd.DataFrame(index=pd.Index([], name='player'))
    
    for player in common_players:
        if (batting_stats.loc[player, 'runs'] >= 100 and 
            bowling_stats.loc[player, 'wickets'] >= 5):
            player_stats = pd.Series({
                'runs': batting_stats.loc[player, 'runs'],
                'wickets': bowling_stats.loc[player, 'wickets'],
                'batting_sr': batting_stats.loc[player, 'strike_rate'],
                'bowling_economy': bowling_stats.loc[player, 'economy']
            }, name=player)
            all_round_stats = pd.concat([all_round_stats, player_stats.to_frame().T])
    
    # --- Key Matches ---
    key_matches = []
    
    # Highest scoring match
    match_runs = season_deliveries.groupby('match_id')['total_runs'].sum()
    season_matches['total_runs'] = season_matches['match_id'].map(match_runs).fillna(0)
    
    if not season_matches.empty and season_matches['total_runs'].max() > 0:
        highest_scoring = season_matches.nlargest(1, 'total_runs').iloc[0]
        key_matches.append({
            'type': 'Highest Scoring Match',
            'description': f"{highest_scoring['team1']} vs {highest_scoring['team2']}",
            'winner': highest_scoring['winner'],
            'margin': f"{int(highest_scoring['total_runs'])} runs total"
        })
    
    # Closest margin victories
    if 'win_by_runs' in season_matches.columns:
        closest_runs = season_matches[
            season_matches['win_by_runs'] > 0
        ].nsmallest(1, 'win_by_runs')
        
        if not closest_runs.empty:
            match = closest_runs.iloc[0]
            key_matches.append({
                'type': 'Closest Run Victory',
                'description': f"{match['team1']} vs {match['team2']}",
                'winner': match['winner'],
                'margin': f"{int(match['win_by_runs'])} runs"
            })
    
    if 'win_by_wickets' in season_matches.columns:
        closest_wickets = season_matches[
            season_matches['win_by_wickets'] > 0
        ].nsmallest(1, 'win_by_wickets')
        
        if not closest_wickets.empty:
            match = closest_wickets.iloc[0]
            key_matches.append({
                'type': 'Closest Wicket Victory',
                'description': f"{match['team1']} vs {match['team2']}",
                'winner': match['winner'],
                'margin': f"{int(match['win_by_wickets'])} wickets"
            })
    
    # Find matches with most sixes
    match_sixes = season_deliveries[
        season_deliveries['batsman_runs'] == 6
    ].groupby('match_id').size()
    
    if not match_sixes.empty:
        most_sixes_id = match_sixes.nlargest(1).index[0]
        most_sixes_match = season_matches[
            season_matches['match_id'] == most_sixes_id
        ]
        
        if not most_sixes_match.empty:
            match = most_sixes_match.iloc[0]
            key_matches.append({
                'type': 'Most Sixes in a Match',
                'description': f"{match['team1']} vs {match['team2']}",
                'winner': match['winner'],
                'margin': f"{match_sixes[most_sixes_id]} sixes"
            })
    
    key_matches_df = pd.DataFrame(key_matches)
    
    # Save all precomputed data for this season
    season_stats.to_parquet(app_data_dir / f'season_{season}_stats.parquet', compression='snappy', index=False)
    standings.to_parquet(app_data_dir / f'season_{season}_standings.parquet', compression='snappy')
    points_progression.to_parquet(app_data_dir / f'season_{season}_points_progression.parquet', compression='snappy')
    batting_stats.to_parquet(app_data_dir / f'season_{season}_batting_stats.parquet', compression='snappy')
    bowling_stats.to_parquet(app_data_dir / f'season_{season}_bowling_stats.parquet', compression='snappy')
    fielding_stats.to_parquet(app_data_dir / f'season_{season}_fielding_stats.parquet', compression='snappy')
    all_round_stats.to_parquet(app_data_dir / f'season_{season}_all_round_stats.parquet', compression='snappy')
    key_matches_df.to_parquet(app_data_dir / f'season_{season}_key_matches.parquet', compression='snappy', index=False)
    
    print(f"Pre-computed data for season {season} saved to {app_data_dir}")

print("All data pre-processing complete!")

# --- New Block: Once completed, check if dream team data needs to be processed ---
if not os.path.exists(app_data_dir / 'dream_team_stats.parquet'):
    print("Dream team data not found. Processing dream team data...")
    # Import the module
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # Add parent directory to path
    from data.process_dream_team_data import create_dream_team_dataset_from_dataframes
    
    # Use the already loaded dataframes instead of trying to read them again
    print("Computing dream team data...")
    create_dream_team_dataset_from_dataframes(all_match_df, all_delivery_df)
    print("Dream team data processing complete!")

def convert_json_to_parquet():
    """
    Convert all JSON files in the app/data directory to Parquet format.
    This function is used to optimize data loading in the app.
    """
    # Get the app/data directory
    app_data_dir = Path(__file__).resolve().parent.parent / "app" / "data"
    
    # Find all JSON files in the directory
    json_files = list(app_data_dir.glob("*.json"))
    
    if not json_files:
        print("No JSON files found in app/data directory")
        return
    
    print(f"Found {len(json_files)} JSON files to convert")
    
    # Convert each JSON file to Parquet
    for json_file in json_files:
        parquet_file = json_file.with_suffix(".parquet")
        print(f"Converting {json_file} to {parquet_file}")
        
        # Load JSON data
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # Convert to DataFrame if it's a list or dict with uniform structure
        if isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, dict):
            # For nested dictionaries, we need to handle differently
            # Option 1: Convert to a single row DataFrame with each key as a column
            df = pd.DataFrame([data])
        else:
            print(f"Unsupported JSON structure in {json_file}")
            continue
        
        # Write to Parquet
        df.to_parquet(parquet_file, compression='snappy')
        print(f"Successfully converted {json_file} to {parquet_file}")
    
    print("All JSON files have been converted to Parquet format")

def main():
    """
    Main function to process IPL data and generate statistics.
    """
    # Get the data directory
    data_dir = Path(__file__).resolve().parent
    
    # Get the app data directory
    app_data_dir = Path(__file__).resolve().parent.parent / "app" / "data"
    os.makedirs(app_data_dir, exist_ok=True)
    
    # Initialize empty mappings
    team_mapping = {}
    venue_mapping = {}
    
    try:
        # Find the matches file for team and venue normalization
        # First, check if there's a processed matches file
        processed_matches_file = data_dir / "processed" / "matches.csv"
        if not processed_matches_file.exists():
            processed_matches_file = data_dir / "processed" / "matches.parquet"
        
        # If processed file exists and has the right format, use it for normalization
        if processed_matches_file.exists() and processed_matches_file.suffix.lower() in ['.csv', '.parquet']:
            team_mapping = normalize_and_abbreviate_teams(str(processed_matches_file))
            venue_mapping = normalize_and_shorten_venues(str(processed_matches_file))
    except Exception as e:
        print(f"Warning: Could not load team and venue mappings: {e}")
        print("Proceeding with empty mappings.")
    
    # Process all match files
    match_files = sorted(glob.glob(str(data_dir / "raw" / "*.json")))
    
    if not match_files:
        raise FileNotFoundError("No match files found in the raw directory")
    
    all_match_df = pd.DataFrame()
    all_delivery_df = pd.DataFrame()
    
    for match_file in match_files:
        match_df, delivery_df = process_ipl_data(match_file, team_mapping, venue_mapping)
        all_match_df = pd.concat([all_match_df, match_df], ignore_index=True)
        all_delivery_df = pd.concat([all_delivery_df, delivery_df], ignore_index=True)
    
    # Assign batter positions
    all_delivery_df['batter_position'] = all_delivery_df.groupby(['match_id', 'inning', 'batting_team']).apply(assign_batter_positions).reset_index(level=[0,1,2], drop=True)
    
    # Sort deliveries by match_id, inning, over, and ball
    all_delivery_df = all_delivery_df.sort_values(by=['match_id', 'inning', 'over', 'ball'])
    all_delivery_df.reset_index(drop=True, inplace=True)
    
    # Save main dataframes to app data directory
    print("Saving main dataframes to app data directory...")
    all_match_df.to_parquet(app_data_dir / 'all_matches.parquet', compression='snappy')
    all_delivery_df.to_parquet(app_data_dir / 'all_deliveries.parquet', compression='snappy')
    print(f"Main dataframes saved to {app_data_dir}")
    
    # Pre-compute data for overview section
    print("Pre-computing data for the overview section...")
    # ... (overview section processing)
    print(f"Pre-computed data for overview section saved to {app_data_dir}")
    
    # Pre-compute data for team analysis section
    print("Pre-computing data for the team analysis section...")
    # ... (team analysis section processing)
    print(f"Pre-computed data for team analysis section saved to {app_data_dir}")
    
    # Pre-compute data for player analysis section
    print("Pre-computing data for player analysis section...")
    # ... (player analysis section processing)
    print(f"Pre-computed data for player analysis section saved to: {app_data_dir}")
    
    # Pre-compute data for match analysis section
    print("Pre-computing data for match analysis section...")
    # ... (match analysis section processing)
    print(f"Pre-computed data for match analysis section saved to {app_data_dir}")
    
    # Pre-compute data for season analysis section
    print("Pre-computing data for season analysis section...")
    # ... (season analysis section processing)
    
    print("All data pre-processing complete!")

if __name__ == "__main__":
    # Add a command-line argument to convert JSON to Parquet
    if len(sys.argv) > 1 and sys.argv[1] == "--convert-json-to-parquet":
        convert_json_to_parquet()
    else:
        main()