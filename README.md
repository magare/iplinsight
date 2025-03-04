# IPL Data Analysis Streamlit App

A beautiful, neon-styled interactive dashboard for exploring Indian Premier League cricket data.

## Features

- 🌈 Neon-themed UI with consistent styling across all charts and components
- 📱 Responsive design that works well on both desktop and mobile
- 📊 Interactive charts and visualizations
- 🏏 Comprehensive analysis of IPL matches, players, teams, and seasons

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

## New UI Features

- **Neon Color Scheme**: The app now uses a cohesive neon color scheme that's visually appealing and consistent across all charts
- **Custom CSS Styling**: Enhanced UI elements with glowing effects and modern styling
- **Improved Navigation**: Streamlined sidebar navigation with better organization
- **Responsive Charts**: Charts now adapt to screen size for better viewing on any device
- **Team-specific Colors**: Each IPL team is represented by a specific neon color for consistent visual identification

## Structure

- `app/` - Main application code
  - `app.py` - Entry point for the Streamlit app
  - `static/` - Static assets including CSS
  - `components/` - Individual analysis components
  - `utils/` - Utility functions and helpers
  - `data/` - Pre-processed data files
