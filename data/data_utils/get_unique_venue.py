import pandas as pd

def get_unique_venues(matches_csv_path):
    """
    Reads the matches.csv file and returns a sorted list of unique venues.
    """
    try:
        matches_df = pd.read_csv(matches_csv_path)
        unique_venues = sorted(matches_df['venue'].unique())
        return unique_venues
    except FileNotFoundError:
        print(f"Error: File not found at {matches_csv_path}")
        return None

# Path to your matches.csv file
matches_csv = '../../data/processed/matches.csv'

# Get the unique venues
unique_venues_list = get_unique_venues(matches_csv)

# Print the unique venues
if unique_venues_list:
    print("Unique Venues:")
    for venue in unique_venues_list:
        print(venue)

    # You can also save the list to a file if needed:
    # with open('../../data/processed/unique_venues.txt', 'w') as f:
    #     for venue in unique_venues_list:
    #         f.write(venue + '\n')
    print("\nUnique venues list saved to ../../data/processed/unique_venues.txt")