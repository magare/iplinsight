"""
Script to update all chart rendering in the app to use responsive_plotly_chart.
Run this script once to update all components.
"""

import os
import re
from pathlib import Path

def update_chart_calls(file_path):
    """
    Update all st.plotly_chart calls to responsive_plotly_chart in a file.
    
    Args:
        file_path: Path to the file to update
    
    Returns:
        bool: True if file was updated, False otherwise
    """
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Check if the file already imports responsive_plotly_chart
    import_pattern = r'from utils\.chart_utils import .*responsive_plotly_chart'
    has_import = bool(re.search(import_pattern, content))
    
    # Add import if needed
    if not has_import:
        if 'from utils.chart_utils import' in content:
            # Append to existing import
            content = re.sub(
                r'from utils\.chart_utils import (.*)',
                r'from utils.chart_utils import \1, responsive_plotly_chart',
                content
            )
        else:
            # Add new import after other imports
            content = re.sub(
                r'(import .*\n)(?!import)',
                r'\1from utils.chart_utils import responsive_plotly_chart\n',
                content,
                count=1
            )
    
    # Replace st.plotly_chart with responsive_plotly_chart
    updated_content = re.sub(
        r'st\.plotly_chart\((.*?)\)',
        r'responsive_plotly_chart(\1)',
        content
    )
    
    # Only write if changes were made
    if updated_content != content:
        with open(file_path, 'w') as f:
            f.write(updated_content)
        return True
    
    return False

def main():
    """Update all component files to use responsive_plotly_chart."""
    # Get the app directory
    app_dir = Path(__file__).resolve().parent.parent
    components_dir = app_dir / 'components'
    
    # List of component files to update
    component_files = [
        components_dir / 'team_analysis.py',
        components_dir / 'player_analysis.py',
        components_dir / 'match_analysis.py',
        components_dir / 'season_analysis.py',
        components_dir / 'venue_analysis.py',
        components_dir / 'dream_team_analysis.py',
        components_dir / 'overview.py'
    ]
    
    # Update each file
    updated_files = []
    for file_path in component_files:
        if file_path.exists():
            if update_chart_calls(file_path):
                updated_files.append(file_path.name)
    
    # Print summary
    if updated_files:
        print(f"Updated {len(updated_files)} files:")
        for file_name in updated_files:
            print(f"  - {file_name}")
    else:
        print("No files needed updating.")

if __name__ == "__main__":
    main() 