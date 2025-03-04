import os
import re
import shutil
from pathlib import Path

# List of core files that must be kept - these are directly loaded by data_loader.py
CORE_FILES = [
    'matches.parquet',
    'deliveries.parquet',
    'dream_team_stats.parquet'  # for dream team analysis
]

# Files needed for overview section
OVERVIEW_FILES = [
    'overview_matches_per_season.parquet',
    'overview_avg_runs_by_season.parquet',
    'overview_avg_wickets_by_season.parquet',
    'overview_team_participation.parquet'
]

# Files needed for team analysis section
TEAM_FILES = [
    'team_stats.parquet',
    'batting_first_stats.parquet',
    'nrr_data.parquet',
    'avg_nrr_by_season.parquet',
    'playoff_stats.parquet',
    'head_to_head_matrix.parquet',
    'venue_team_stats.parquet',
    'team_phase_stats.parquet'
]

# Files needed for player analysis section
PLAYER_FILES = [
    'player_batting_stats.parquet',
    'player_batting_phase_stats.parquet',
    'player_position_stats.parquet',
    'player_bowling_stats.parquet',
    'player_wicket_types.parquet',
    'player_bowling_phase_stats.parquet',
    'player_allrounder_stats.parquet',
    'player_h2h_stats.parquet'
]

# Files needed for match analysis section
MATCH_FILES = [
    'match_runs_victories.parquet',
    'match_wickets_victories.parquet',
    'match_win_method_by_season.parquet',
    'match_win_method_by_season_pct.parquet',
    'match_toss_decisions.parquet',
    'match_toss_by_season.parquet',
    'match_toss_by_season_pct.parquet',
    'match_toss_by_venue.parquet',
    'match_toss_by_venue_pct.parquet',
    'match_toss_win_pct.parquet',
    'match_toss_decision_outcomes.parquet',
    'match_scores.parquet',
    'match_venue_scores.parquet',
    'match_season_scores.parquet',
    'match_phase_avg.parquet',
    'match_score_thresholds.parquet',
    'match_high_scoring_venues.parquet',
    'match_low_scoring_venues.parquet',
    'match_high_scoring_teams.parquet',
    'match_low_scoring_teams.parquet',
    'match_high_scoring_seasons.parquet',
    'match_low_scoring_seasons.parquet',
    'match_high_scoring_toss.parquet',
    'match_low_scoring_toss.parquet',
    'match_high_scoring_phases.parquet',
    'match_low_scoring_phases.parquet'
]

# Files needed for season analysis section - basic files only, we'll handle pattern matching for season-specific files
SEASON_BASE_FILES = []

# Create a regex pattern for season-specific files that we need to keep
SEASON_FILE_PATTERNS = [
    r'season_\d{4}_stats\.parquet',
    r'season_\d{4}_standings\.parquet',
    r'season_\d{4}_points_progression\.parquet',
    r'season_\d{4}_batting_stats\.parquet',
    r'season_\d{4}_bowling_stats\.parquet',
    r'season_\d{4}_all_round_stats\.parquet',
    r'season_\d{4}_fielding_stats\.parquet',
    r'season_\d{4}_key_matches\.parquet'
]

def main():
    """Clean up unnecessary files from the app data directory."""
    # Get the current directory and app data directory
    app_data_dir = Path('app/data')
    
    if not app_data_dir.exists():
        print(f"Error: Data directory {app_data_dir} not found!")
        return
    
    # Create a backup directory
    backup_dir = Path('app/data_backup')
    if not backup_dir.exists():
        backup_dir.mkdir()
        print(f"Created backup directory: {backup_dir}")
    
    # Combine all required files
    required_files = CORE_FILES + OVERVIEW_FILES + TEAM_FILES + PLAYER_FILES + MATCH_FILES + SEASON_BASE_FILES
    
    # Get all .parquet files in the data directory
    all_files = [f.name for f in app_data_dir.glob('*.parquet')]
    print(f"Found {len(all_files)} Parquet files in {app_data_dir}")
    
    # Identify required season-specific files
    required_season_files = []
    for file in all_files:
        for pattern in SEASON_FILE_PATTERNS:
            if re.match(pattern, file):
                required_season_files.append(file)
                break
    
    # Combine all required files
    required_files = required_files + required_season_files
    print(f"Identified {len(required_files)} essential Parquet files to keep")
    
    # Identify files to remove
    files_to_remove = []
    for file in all_files:
        if file not in required_files:
            files_to_remove.append(file)
    
    print(f"Found {len(files_to_remove)} unnecessary Parquet files that can be removed")
    
    # Move unnecessary files to the backup directory
    if files_to_remove:
        print("Moving unnecessary files to backup directory...")
        for file in files_to_remove:
            src_path = app_data_dir / file
            dst_path = backup_dir / file
            shutil.move(src_path, dst_path)
            print(f"  Moved: {file}")
        
        print(f"Successfully moved {len(files_to_remove)} unnecessary files to {backup_dir}")
        print("You can delete the backup directory if you're sure you don't need these files.")
    else:
        print("No unnecessary files found. All files in the data directory are required.")

if __name__ == "__main__":
    main() 