import pandas as pd
import os
from pathlib import Path

def convert_csv_to_parquet(csv_path):
    """
    Convert a CSV file to Parquet format.
    
    Args:
        csv_path (str): Path to the CSV file
    """
    # Read the CSV file
    try:
        # Try reading with index_col=0 first (for files that have an index column)
        df = pd.read_csv(csv_path, index_col=0)
        # Check if the index is meaningful (not just row numbers)
        if df.index.name is None and df.index.dtype == 'int64' and (df.index == range(len(df))).all():
            # If it's just row numbers, read without index_col
            df = pd.read_csv(csv_path)
    except:
        # If that fails, try reading without index_col
        df = pd.read_csv(csv_path)
    
    # Create the Parquet file path
    parquet_path = csv_path.replace('.csv', '.parquet')
    
    # Write to Parquet
    df.to_parquet(parquet_path, index=True)
    
    print(f"Converted {csv_path} to {parquet_path}")

def main():
    # List of missing Parquet files
    missing_files = [
        "app/data/dream_team_stats.csv",
        "app/data/season_2008_points_progression.csv",
        "app/data/season_2009_points_progression.csv",
        "app/data/season_2010_points_progression.csv",
        "app/data/season_2011_points_progression.csv",
        "app/data/season_2012_points_progression.csv",
        "app/data/season_2013_points_progression.csv",
        "app/data/season_2014_points_progression.csv",
        "app/data/season_2015_points_progression.csv",
        "app/data/season_2016_points_progression.csv",
        "app/data/season_2017_points_progression.csv",
        "app/data/season_2018_points_progression.csv",
        "app/data/season_2019_points_progression.csv",
        "app/data/season_2020_points_progression.csv",
        "app/data/season_2021_points_progression.csv",
    ]
    
    # Convert each file
    for file_path in missing_files:
        if os.path.exists(file_path):
            convert_csv_to_parquet(file_path)
        else:
            print(f"File not found: {file_path}")

if __name__ == "__main__":
    main() 