# IPL Data Processing

This directory contains scripts and data for processing IPL (Indian Premier League) cricket data.

## Directory Structure

- `raw/`: Contains raw data files (JSON format) downloaded from sources
- `processed/`: Contains processed data files (Parquet format) ready for analysis
- `process_data.py`: Main script for processing raw data into processed format

## Data Processing Workflow

1. Download raw IPL match data in JSON format and place in the `raw/` directory
2. Run the processing script: `python process_data.py`
3. Processed data will be saved in the `processed/` directory
4. The app will use the processed data from the `app/data/` directory

## Data Sources

The IPL data can be obtained from various sources:

- [Cricsheet](https://cricsheet.org/): Provides ball-by-ball data in YAML/JSON format
- [IPL Official Website](https://www.iplt20.com/): Official statistics and match data
- [Kaggle IPL Datasets](https://www.kaggle.com/datasets?search=ipl): Various IPL datasets

## Data Format

The processed data follows these formats:

### Matches Data

Contains match-level information like teams, venue, toss details, etc.

### Deliveries Data

Contains ball-by-ball information including batsman, bowler, runs, wickets, etc.
