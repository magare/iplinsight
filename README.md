# IPL Data Analysis Streamlit App

A beautiful, neon-styled interactive dashboard for exploring Indian Premier League cricket data.

## Features

- 🌈 Neon-themed UI with consistent styling across all charts and components
- 📱 Responsive design that works well on both desktop and mobile
- 📊 Interactive charts and visualizations
- 🏏 Comprehensive analysis of IPL matches, players, teams, and seasons

## Project Structure

- `app/` - Main application code
  - `app.py` - Entry point for the Streamlit app
  - `static/` - Static assets including CSS
  - `components/` - Individual analysis components
  - `utils/` - Utility functions and helpers
  - `data/` - Pre-processed data files used by the app
- `data/` - Data processing scripts and raw data
  - `raw/` - Raw data files (JSON format)
  - `processed/` - Processed data files (Parquet format)
  - `process_data.py` - Main script for data processing

## Data Processing Workflow

1. Download raw IPL match data in JSON format and place in the `data/raw/` directory
2. Run the processing script:
   ```
   python data/process_data.py
   ```
3. The processed data will be automatically copied to the `app/data/` directory

## Running the App

1. Navigate to the app directory:

   ```
   cd streamlit_app
   ```

2. Install dependencies (if not already installed):

   ```
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:

   ```
   streamlit run app/app.py
   ```

4. Open your browser and go to the URL shown in the terminal (typically http://localhost:8501)

## UI Features

- **Neon Color Scheme**: The app uses a cohesive neon color scheme that's visually appealing and consistent across all charts
- **Custom CSS Styling**: Enhanced UI elements with glowing effects and modern styling
- **Improved Navigation**: Streamlined sidebar navigation with better organization
- **Responsive Charts**: Charts adapt to screen size for better viewing on any device
- **Team-specific Colors**: Each IPL team is represented by a specific neon color for consistent visual identification

## Data Sources

The IPL data can be obtained from various sources:

- [Cricsheet](https://cricsheet.org/): Provides ball-by-ball data in YAML/JSON format
- [IPL Official Website](https://www.iplt20.com/): Official statistics and match data
- [Kaggle IPL Datasets](https://www.kaggle.com/datasets?search=ipl): Various IPL datasets
