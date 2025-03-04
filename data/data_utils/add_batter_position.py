import pandas as pd
from pathlib import Path
import random

def add_batter_positions(deliveries_csv_path, output_path=None):
    """
    Add batter position column to the deliveries data.
    Position is determined by the order in which batters come to bat in each innings.
    Openers are positions 1 and 2, subsequent batters get positions 3 onwards.
    
    Args:
        deliveries_csv_path: Path to the deliveries.csv file
        output_path: Optional path to save the output. If None, will overwrite input file
    
    Returns:
        DataFrame with added batter_position column
    """
    # Read the deliveries data
    df = pd.read_csv(deliveries_csv_path)
    
    # Initialize the batter_position column
    df['batter_position'] = 0
    
    # Process each match and innings separately
    for match_id in df['match_id'].unique():
        match_data = df[df['match_id'] == match_id]
        
        for inning in [1, 2]:
            innings_data = match_data[match_data['inning'] == inning]
            
            if innings_data.empty:
                continue
                
            # Get unique batters in order of appearance
            batters = innings_data['batter'].unique()
            
            # Create a dictionary mapping batters to their positions
            batter_positions = {}
            for position, batter in enumerate(batters, start=1):
                batter_positions[batter] = position
            
            # Update positions in the main dataframe
            mask = (df['match_id'] == match_id) & (df['inning'] == inning)
            df.loc[mask, 'batter_position'] = df.loc[mask, 'batter'].map(batter_positions)
    
    # Save the updated data if output path is provided
    if output_path:
        df.to_csv(output_path, index=False)
    else:
        df.to_csv(deliveries_csv_path, index=False)
    
    return df

def verify_match(df, match_id):
    """Print verification info for a specific match."""
    print(f"\nVerifying match {match_id}:")
    match_data = df[df['match_id'] == match_id]
    
    for inning in [1, 2]:
        print(f"\nInning {inning}:")
        innings_data = match_data[match_data['inning'] == inning]
        if innings_data.empty:
            continue
            
        # Show batters in order of appearance with their positions
        batter_info = innings_data[['batter', 'batter_position']].drop_duplicates()
        batter_info = batter_info.sort_values('batter_position')
        
        # Add information about how each batter got out
        wickets = innings_data[innings_data['is_wicket'] == 1][['player_out', 'wicket_kind']].drop_duplicates()
        
        print("\nBatters and their positions:")
        for _, row in batter_info.iterrows():
            out_info = wickets[wickets['player_out'] == row['batter']]
            out_str = f" (Out: {out_info.iloc[0]['wicket_kind']})" if not out_info.empty else ""
            print(f"Position {row['batter_position']}: {row['batter']}{out_str}")
            
        print("\nWickets in order:")
        wicket_deliveries = innings_data[innings_data['is_wicket'] == 1].sort_values(['over', 'ball'])
        for _, row in wicket_deliveries.iterrows():
            print(f"Over {row['over']}.{row['ball']}: {row['player_out']} out ({row['wicket_kind']})")

if __name__ == '__main__':
    # Get project root directory
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    
    # Define paths
    deliveries_path = project_root / 'data' / 'processed' / 'deliveries.csv'
    
    # Add batter positions
    updated_df = add_batter_positions(deliveries_path)
    
    # Print verification info for first match
    first_match_id = updated_df['match_id'].iloc[0]
    verify_match(updated_df, first_match_id)
    
    # Print verification info for a random match
    all_matches = updated_df['match_id'].unique()
    random_match = random.choice(all_matches)
    verify_match(updated_df, random_match) 