import pandas as pd
import json
import os
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq

def convert_json_to_parquet(json_file_path, parquet_file_path):
    """
    Convert a JSON file to Parquet format.
    
    Args:
        json_file_path (str): Path to the JSON file
        parquet_file_path (str): Path to save the Parquet file
    """
    print(f"Converting {json_file_path} to {parquet_file_path}")
    
    # Load JSON data
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    # Convert to DataFrame if it's a list or dict with uniform structure
    if isinstance(data, list):
        df = pd.DataFrame(data)
    elif isinstance(data, dict):
        # For nested dictionaries, we need to handle differently
        # Option 1: Convert to a single row DataFrame with each key as a column
        df = pd.DataFrame([data])
    else:
        raise ValueError(f"Unsupported JSON structure in {json_file_path}")
    
    # Write to Parquet
    df.to_parquet(parquet_file_path)
    print(f"Successfully converted {json_file_path} to {parquet_file_path}")

def main():
    # Get the app/data directory
    data_dir = Path("app/data")
    
    # Find all JSON files in the directory
    json_files = list(data_dir.glob("*.json"))
    
    if not json_files:
        print("No JSON files found in app/data directory")
        return
    
    print(f"Found {len(json_files)} JSON files to convert")
    
    # Convert each JSON file to Parquet
    for json_file in json_files:
        parquet_file = json_file.with_suffix(".parquet")
        convert_json_to_parquet(json_file, parquet_file)
    
    print("All JSON files have been converted to Parquet format")

if __name__ == "__main__":
    main() 