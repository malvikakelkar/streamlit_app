import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import json
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import logging
import gc
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Basic page config
st.set_page_config(
    layout="wide",
    page_title="Asteroid Trajectory",
    page_icon="☄️",
)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'data' not in st.session_state:
    st.session_state.data = {}
if 'selected_asteroid' not in st.session_state:
    st.session_state.selected_asteroid = 16
if 'show_planets' not in st.session_state:
    st.session_state.show_planets = {
        'Mercury': True,
        'Venus': True,
        'Earth': True,
        'Mars': True,
        'Jupiter': True
    }

# Utility functions
def format_mass(mass_str):
    """Format mass values with units"""
    try:
        mass = float(mass_str)
        return f"{mass:.2e} kg"
    except ValueError:
        return mass_str

def clean_memory():
    """Memory management utility"""
    gc.collect()

# Data loading functions
@st.cache_data(ttl=3600)
def load_asteroid_info():
    """Load asteroid information with dummy data for testing"""
    try:
        with open('asteroid_info.json', 'r') as f:
            return json.load(f)
    except:
        # Return dummy data for testing
        return {
            "16": {
                "name": "Psyche",
                "type": "M",
                "group": "Main Belt",
                "orbital_period": "1825",
                "mass": "2.72e19",
                "diameter": "226",
                "aphelion": "3.328",
                "perihelion": "2.513",
                "eccentricity": "0.139"
            }
        }

def load_dummy_trajectory_data():
    """Create dummy trajectory data for testing"""
    t = np.linspace(0, 10, 100)
    return pd.DataFrame({
        'time': t,
        'x': np.sin(t),
        'y': np.cos(t),
        'z': t * 0.1
    })

def lazy_load_data(asteroid_id):
    """Load data with dummy data fallback"""
    if not st.session_state.data_loaded:
        # For testing, load dummy data
        dummy_data = load_dummy_trajectory_data()
        st.session_state.data = {
            'actual': dummy_data,
            'pinn': dummy_data,
            'nbody': dummy_data,
            'transformer': dummy_data,
            'lstm': dummy_data
        }
        st.session_state.data_loaded = True
    return st.session_state.data

# Display functions
def display_overview():
    """Display overview tab content"""
    st.header("Overview")
    
    st.write("""
    ### About This Dashboard
    This dashboard visualizes asteroid trajectories using different prediction models:
    - Physics-Informed Neural Networks (PINN)
    - N-body Simulation
    - Transformer Model
    - LSTM Model
    """)
    
    # Display some dummy metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Asteroids", "9")
    with col2:
        st.metric("Time Period", "2000-2024")
    with col3:
        st.metric("Prediction Models", "4")

def display_actual_trajectory():
    """Display actual trajectory tab content"""
    st.header("Actual Trajectory")
    
    if 'actual' in st.session_state.data:
        data = st.session_state.data['actual']
        
        # Create a simple 3D scatter plot
        fig = go.Figure(data=[go.Scatter3d(
            x=data['x'],
            y=data['y'],
            z=data['z'],
            mode='lines',
            line=dict(color='blue', width=2)
        )])
        
        fig.update_layout(
            scene=dict(
                xaxis_title='X (AU)',
                yaxis_title='Y (AU)',
                zaxis_title='Z (AU)'
            ),
            height=700
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No trajectory data available")

def display_model_trajectories():
    """Display model trajectories tab content"""
    st.header("Model Trajectories")
    
    # Model selection
    model = st.selectbox(
        "Select Model",
        ["PINN", "N-body", "Transformer", "LSTM"]
    )
    
    if st.session_state.data:
        data = st.session_state.data.get(model.lower())
        if data is not None:
            fig = go.Figure(data=[go.Scatter3d(
                x=data['x'],
                y=data['y'],
                z=data['z'],
                mode='lines',
                line=dict(color='red', width=2)
            )])
            
            fig.update_layout(
                scene=dict(
                    xaxis_title='X (AU)',
                    yaxis_title='Y (AU)',
                    zaxis_title='Z (AU)'
                ),
                height=700
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning(f"No data available for {model} model")
    else:
        st.warning("No model data available")

# Main app function
def main():
    """Main function to run the dashboard"""
    # Header
    st.title("☄️ Asteroid Trajectory Dashboard")
    
    # Available asteroid IDs
    asteroid_ids = [16, 21, 243, 253, 433, 4179, 25143, 101955, 162173]
    
    # Asteroid selection
    selected_asteroid = st.selectbox(
        "Select Asteroid",
        options=asteroid_ids,
        index=asteroid_ids.index(st.session_state.selected_asteroid)
    )
    
    # Update selected asteroid
    if selected_asteroid != st.session_state.selected_asteroid:
        st.session_state.selected_asteroid = selected_asteroid
        st.session_state.data_loaded = False
    
    # Load asteroid info and data
    asteroid_info = load_asteroid_info()
    data = lazy_load_data(selected_asteroid)
    
    # Display asteroid information
    if str(selected_asteroid) in asteroid_info:
        info = asteroid_info[str(selected_asteroid)]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Name", info['name'])
        with col2:
            st.metric("Type", info['type'])
        with col3:
            st.metric("Diameter", f"{info['diameter']} km")
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Overview", "Actual Trajectory", "Model Trajectories"])
    
    with tab1:
        display_overview()
    with tab2:
        display_actual_trajectory()
    with tab3:
        display_model_trajectories()

# Run the app
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        logger.exception("Application error")