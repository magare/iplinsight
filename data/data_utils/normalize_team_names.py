import pandas as pd
from pathlib import Path
import os

def normalize_and_abbreviate_teams(file_path):
    """
    Reads the matches file (CSV or Parquet), normalizes team names based on manual mapping,
    abbreviates them, and returns a dictionary mapping original names to normalized/abbreviated names.

    Args:
        file_path: Path to the matches file (CSV or Parquet).

    Returns:
        A dictionary where keys are original team names and values are normalized/abbreviated names.
    """
    # Determine the file type and load data accordingly
    file_path = Path(file_path)
    
    if file_path.suffix.lower() == '.csv':
        matches_df = pd.read_csv(file_path)
    elif file_path.suffix.lower() == '.parquet':
        matches_df = pd.read_parquet(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}. Only CSV and Parquet are supported.")

    # Load the matches data and extract unique team names
    team1_teams = matches_df['team1'].unique()
    team2_teams = matches_df['team2'].unique()
    all_teams = pd.concat([pd.Series(team1_teams), pd.Series(team2_teams)]).unique()
    unique_teams = sorted(all_teams)

    # Manual Mapping (with Normalization and Abbreviation)
    manual_mapping = {
        'Chennai Super Kings': 'CSK',
        'Deccan Chargers': 'SRH',  # Renamed Deccan Chargers to Sunrisers Hyderabad
        'Delhi Capitals': 'DC',
        'Delhi Daredevils': 'DD',
        'Gujarat Lions': 'GL',
        'Gujarat Titans': 'GT',
        'Kings XI Punjab': 'KXIP',
        'Kochi Tuskers Kerala': 'KTK',
        'Kolkata Knight Riders': 'KKR',
        'Lucknow Super Giants': 'LSG',
        'Mumbai Indians': 'MI',
        'Pune Warriors': 'PW',
        'Punjab Kings': 'PBKS',  # Different from KXIP
        'Rajasthan Royals': 'RR',
        'Rising Pune Supergiant': 'RPS',
        'Rising Pune Supergiants': 'RPS',
        'Royal Challengers Bangalore': 'RCB',
        'Royal Challengers Bengaluru': 'RCB',
        'Sunrisers Hyderabad': 'SRH'
    }

    # Create a mapping from original names to normalized/abbreviated names
    team_mapping = {}
    for team in unique_teams:
        if team in manual_mapping:
            team_mapping[team] = manual_mapping[team]
        else:
            team_mapping[team] = team  # No mapping found, keep the original name

    return team_mapping

if __name__ == '__main__':
    # This code only runs when the file is executed directly
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    matches_file_path = project_root / 'data' / 'processed' / 'matches.csv'
    output_format = 'parquet'  # Change to 'csv' if needed
    
    # Get the team mapping
    team_mapping = normalize_and_abbreviate_teams(str(matches_file_path))
    
    # Print the mapping
    print("Team Mapping (Original -> Normalized/Abbreviated):")
    for original, normalized in team_mapping.items():
        print(f"  '{original}' -> '{normalized}'")
    
    # Apply the mapping to the DataFrame if needed
    if matches_file_path.suffix.lower() == '.csv':
        matches_df = pd.read_csv(matches_file_path)
    else:
        matches_df = pd.read_parquet(matches_file_path)
    
    matches_df['team1'] = matches_df['team1'].map(team_mapping)
    matches_df['team2'] = matches_df['team2'].map(team_mapping)
    if 'toss_winner' in matches_df.columns:
        matches_df['toss_winner'] = matches_df['toss_winner'].map(team_mapping)
    if 'winner' in matches_df.columns:
        matches_df['winner'] = matches_df['winner'].map(team_mapping)
    
    # Save normalized data
    if output_format == 'csv':
        matches_df.to_csv(project_root / 'data' / 'processed' / 'matches_normalized.csv', index=False)
        print("\nNormalized team names applied to matches data and saved to 'data/processed/matches_normalized.csv'")
    else:
        matches_df.to_parquet(project_root / 'data' / 'processed' / 'matches_normalized.parquet', compression='snappy', index=False)
        print("\nNormalized team names applied to matches data and saved to 'data/processed/matches_normalized.parquet'")