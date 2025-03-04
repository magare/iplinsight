import pandas as pd
from pathlib import Path
import os

def normalize_and_shorten_venues(file_path):
    """
    Reads the matches file (CSV or Parquet), applies manual mapping to normalize and shorten venue names,
    and returns a dictionary mapping original names to normalized/shortened names.

    Args:
        file_path: Path to the matches file (CSV or Parquet).

    Returns:
        A dictionary where keys are original venue names and values are normalized/shortened names.
    """
    # Determine the file type and load data accordingly
    file_path = Path(file_path)
    
    if file_path.suffix.lower() == '.csv':
        matches_df = pd.read_csv(file_path)
    elif file_path.suffix.lower() == '.parquet':
        matches_df = pd.read_parquet(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}. Only CSV and Parquet are supported.")

    # Load the matches data and extract unique venues
    unique_venues = sorted(matches_df['venue'].unique())

    # Manual Mapping (with Shortened Names)
    manual_mapping = {
        'Arun Jaitley Stadium, Delhi': 'Arun Jaitley Stadium',
        'Feroz Shah Kotla': 'Arun Jaitley Stadium',
        'M Chinnaswamy Stadium, Bengaluru': 'M. Chinnaswamy Stadium',
        'M.Chinnaswamy Stadium': 'M. Chinnaswamy Stadium',
        'MA Chidambaram Stadium, Chepauk': 'MA Chidambaram Stadium',
        'MA Chidambaram Stadium, Chepauk, Chennai': 'MA Chidambaram Stadium',
        'Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium, Visakhapatnam': 'ACA-VDCA Stadium',
        'Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium': 'ACA-VDCA Stadium',
        'Eden Gardens, Kolkata': 'Eden Gardens',
        'Rajiv Gandhi International Stadium, Uppal': 'Rajiv Gandhi Stadium',
        'Rajiv Gandhi International Stadium, Hyderabad': 'Rajiv Gandhi Stadium',
        'Punjab Cricket Association IS Bindra Stadium, Mohali': 'PCA Stadium',
        'Punjab Cricket Association Stadium, Mohali': 'PCA Stadium',
        'Maharashtra Cricket Association Stadium, Pune': 'MCA Stadium',
        'Wankhede Stadium, Mumbai': 'Wankhede Stadium',
        'Sardar Patel Stadium, Motera': 'Narendra Modi Stadium',
        'Narendra Modi Stadium, Ahmedabad': 'Narendra Modi Stadium',
        'Dubai International Cricket Stadium': 'Dubai Stadium',
        'Sharjah Cricket Stadium': 'Sharjah Stadium',
        'Sheikh Zayed Stadium, Abu Dhabi': 'Zayed Stadium',
        'Zayed Cricket Stadium, Abu Dhabi': 'Zayed Stadium',
        'M. A. Chidambaram Stadium': 'MA Chidambaram Stadium',
        'Sawai Mansingh Stadium, Jaipur': 'Sawai Mansingh Stadium',
        'Holkar Cricket Stadium, Indore': 'Holkar Stadium',
        'JSCA International Stadium Complex, Ranchi': 'JSCA Stadium',
        'Saurashtra Cricket Association Stadium, Rajkot': 'SCA Stadium',
        'Green Park, Kanpur': 'Green Park Stadium'
    }

    # Create a mapping from original names to normalized/shortened names
    venue_mapping = {}
    for venue in unique_venues:
        if venue in manual_mapping:
            venue_mapping[venue] = manual_mapping[venue]
        else:
            venue_mapping[venue] = venue  # No mapping found, keep the original name

    return venue_mapping

if __name__ == '__main__':
    # This code only runs when the file is executed directly
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    matches_file_path = project_root / 'data' / 'processed' / 'matches.csv'
    output_format = 'parquet'  # Change to 'csv' if needed
    
    # Get the venue mapping
    venue_mapping = normalize_and_shorten_venues(str(matches_file_path))
    
    # Print the mapping
    print("Venue Mapping (Original -> Normalized/Shortened):")
    for original, normalized in venue_mapping.items():
        print(f"  '{original}' -> '{normalized}'")
    
    # Apply the mapping to the DataFrame if needed
    if matches_file_path.suffix.lower() == '.csv':
        matches_df = pd.read_csv(matches_file_path)
    else:
        matches_df = pd.read_parquet(matches_file_path)
    
    matches_df['venue'] = matches_df['venue'].map(venue_mapping)
    
    # Save normalized data
    if output_format == 'csv':
        matches_df.to_csv(project_root / 'data' / 'processed' / 'matches_normalized.csv', index=False)
        print("\nNormalized venue names applied to matches data and saved to 'data/processed/matches_normalized.csv'")
    else:
        matches_df.to_parquet(project_root / 'data' / 'processed' / 'matches_normalized.parquet', compression='snappy', index=False)
        print("\nNormalized venue names applied to matches data and saved to 'data/processed/matches_normalized.parquet'")