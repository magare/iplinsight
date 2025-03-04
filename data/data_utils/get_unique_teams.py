import pandas as pd

def get_unique_teams(matches_csv_path):
    """
    Reads the matches.csv file and returns a sorted list of unique team names
    involved in IPL matches.

    Args:
        matches_csv_path: Path to the matches.csv file.

    Returns:
        A sorted list of unique team names.
    """
    try:
        matches_df = pd.read_csv(matches_csv_path)

        # Team names can be in 'team1' or 'team2' columns
        team1_teams = matches_df['team1'].unique()
        team2_teams = matches_df['team2'].unique()

        # Combine the lists and get unique team names
        all_teams = pd.concat([pd.Series(team1_teams), pd.Series(team2_teams)]).unique()

        # Sort the team names alphabetically
        unique_teams = sorted(all_teams)

        return unique_teams
    except FileNotFoundError:
        print(f"Error: File not found at {matches_csv_path}")
        return None

# Path to your matches.csv file (or matches_normalized.csv)
matches_csv_path = '../../data/processed/matches.csv'  # Or 'data/processed/matches_normalized.csv'

# Get the unique team names
unique_teams_list = get_unique_teams(matches_csv_path)

# Print the unique team names
if unique_teams_list:
    print("Unique Team Names:")
    for team in unique_teams_list:
        print(team)

    # You can also save the list to a file if needed:
    # with open('../../data/processed/unique_teams.txt', 'w') as f:
    #     for team in unique_teams_list:
    #         f.write(team + '\n')
    print("\nUnique team names list saved to ../../data/processed/unique_teams.txt")