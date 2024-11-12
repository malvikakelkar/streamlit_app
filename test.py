import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import json
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from plotly.subplots import make_subplots
import logging
import gc

# Page Configuration
st.set_page_config(
    layout="wide",
    page_title="Asteroid Trajectory",
    page_icon="☄️",
    initial_sidebar_state="auto"
)

# Configure logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
    st.session_state.data = {}
    st.session_state.last_asteroid = None

# Initialize visualization states
if 'show_planets' not in st.session_state:
    st.session_state.show_planets = {
        'Mercury': True,
        'Venus': True,
        'Earth': True,
        'Mars': True,
        'Jupiter': True
    }
if 'current_model' not in st.session_state:
    st.session_state.current_model = 'PINN'
if 'current_coordinate' not in st.session_state:
    st.session_state.current_coordinate = 'x'

# Utility Functions
def clean_memory():
    """Memory management utility"""
    gc.collect()

def julian_to_gregorian(julian_date):
    """Convert Julian date to Gregorian date"""
    return datetime.fromordinal(int(julian_date - 1721425.5)) + timedelta(days=julian_date % 1)

def format_number(num):
    """Format numbers with scientific notation"""
    try:
        return f"{float(num):.5e}"
    except (ValueError, TypeError):
        return "N/A"

def format_mass(mass_str):
    """Format mass values with units"""
    try:
        mass = float(mass_str)
        return f"{mass:.2e} kg"
    except ValueError:
        return mass_str


# Data Loading Functions
@st.cache_data(ttl=3600)
def load_asteroid_info():
    """Load asteroid information from JSON file"""
    try:
        with open('asteroid_info.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error("asteroid_info.json file not found")
        return {}
    except Exception as e:
        logger.error(f"Error loading asteroid info: {str(e)}")
        return {}

@st.cache_data(ttl=3600)
def load_trajectory_data(asteroid_id, data_type):
    """Load trajectory data for a specific asteroid and model"""
    if data_type == 'actual':
        path = Path(f'actual_data/asteroid_{asteroid_id}/asteroid_{asteroid_id}_data.csv')
    else:
        model_dirs = {
            'N-body': ('N-body', 'n_body_predicted_trajectory.csv'),
            'PINN': ('PINN', 'pinn_predicted_trajectory.csv'),
            'Transformer': ('Transformer', 'transformer_predicted_trajectory.csv')
        }
        if data_type not in model_dirs:
            logger.error(f"Unknown data type: {data_type}")
            return None
        model_dir, file_name = model_dirs[data_type]
        path = Path(f'model_data/{model_dir}/asteroid_{asteroid_id}/{file_name}')
    
    try:
        if not path.exists():
            logger.warning(f"File not found: {path}")
            return None
        return pd.read_csv(path)
    except Exception as e:
        logger.error(f"Error loading trajectory data: {str(e)}")
        return None

@st.cache_data(ttl=3600)
def load_trajectory_differences(asteroid_id, model):
    """Load trajectory differences for a specific asteroid and model"""
    model_files = {
        'N-body': 'n_body_trajectory_differences.csv',
        'PINN': 'pinn_trajectory_differences.csv',
        'Transformer': 'transformer_trajectory_differences.csv'
    }
    
    if model not in model_files:
        logger.error(f"Unknown model: {model}")
        return None

    path = Path(f'model_data/{model}/asteroid_{asteroid_id}/{model_files[model]}')
    try:
        if not path.exists():
            logger.warning(f"File not found: {path}")
            return None
        data = pd.read_csv(path)
        expected_columns = ['time', 'x_diff', 'y_diff', 'z_diff', 'vx_diff', 'vy_diff', 'vz_diff']
        if not all(col in data.columns for col in expected_columns):
            logger.error(f"Invalid data structure for {model} model")
            return None
        return data
    except Exception as e:
        logger.error(f"Error loading trajectory differences: {str(e)}")
        return None

@st.cache_data(ttl=3600)
def load_planet_data(asteroid_id, planet):
    """Load planetary data for visualization"""
    try:
        path = Path(f'actual_data/asteroid_{asteroid_id}/{planet}_data.csv')
        return pd.read_csv(path)
    except Exception as e:
        logger.error(f"Error loading planet data: {str(e)}")
        return None

@st.cache_data(ttl=3600)
def load_analysis_summary(asteroid_id, model):
    """Load analysis summary for performance metrics"""
    model_files = {
        "N-body": "n_body_analysis_summary.json",
        "PINN": "pinn_analysis_summary.json",
        "Transformer": "transformer_analysis_summary.json"
    }
    
    if model not in model_files:
        logger.error(f"Unknown model: {model}")
        return None

    file_path = Path(f'model_data/{model}/asteroid_{asteroid_id}/{model_files[model]}')
    try:
        if not file_path.exists():
            logger.warning(f"File not found: {file_path}")
            return None
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading analysis summary: {str(e)}")
        return None

def lazy_load_data(asteroid_id):
    """Lazy loading implementation for asteroid data"""
    if not st.session_state.data_loaded or st.session_state.get('last_asteroid_id') != asteroid_id:
        with st.spinner('Loading asteroid data...'):
            data = {
                'asteroid_id': asteroid_id,
                'actual': load_trajectory_data(asteroid_id, 'actual'),
                'pinn': load_trajectory_data(asteroid_id, 'PINN'),
                'nbody': load_trajectory_data(asteroid_id, 'N-body'),
                'transformer': load_trajectory_data(asteroid_id, 'Transformer'),
                'planets': {
                    planet: load_planet_data(asteroid_id, planet) 
                    for planet in ['Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter']
                },
                'differences': {
                    'pinn': load_trajectory_differences(asteroid_id, 'PINN'),
                    'nbody': load_trajectory_differences(asteroid_id, 'N-body'),
                    'transformer': load_trajectory_differences(asteroid_id, 'Transformer')
                }
            }
            
            if all(v is not None for v in [data['actual'], data['pinn'], data['nbody'], data['transformer']]):
                st.session_state.data = data
                st.session_state.data_loaded = True
                st.session_state.last_asteroid_id = asteroid_id
            else:
                st.error("Error loading some trajectory data. Please try again.")
                return None
                
    return st.session_state.data

def display_trajectory_plot(data, title, is_actual=False, camera_preset="front"):
    """Helper function to display 3D trajectory plot with front view default"""
    fig = go.Figure()
    
    # Add Sun
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0],
        mode='markers',
        marker=dict(size=10, color='yellow'),
        name='Sun'
    ))
    
    # Add trajectory
    fig.add_trace(go.Scatter3d(
        x=data['x'],
        y=data['y'],
        z=data['z'],
        mode='lines',
        name=f"{'Actual' if is_actual else title} Trajectory",
        line=dict(color='blue' if is_actual else 'magenta', width=4)
    ))
    
    # Add planets if enabled
    planet_colors = {
        'Mercury': 'purple',
        'Venus': 'orange',
        'Earth': 'blue',
        'Mars': 'red',
        'Jupiter': 'brown'
    }
    
    for planet, color in planet_colors.items():
        if st.session_state.show_planets.get(planet, True) and planet in st.session_state.data['planets']:
            planet_data = st.session_state.data['planets'][planet]
            if planet_data is not None:
                fig.add_trace(go.Scatter3d(
                    x=planet_data['x'],
                    y=planet_data['y'],
                    z=planet_data['z'],
                    mode='lines',
                    name=f'{planet} Orbit',
                    line=dict(color=color, width=2)
                ))
    
    # Set fixed camera angle for front view
    camera = dict(
        up=dict(x=0, y=0, z=1),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=2, y=-2, z=1.5)  # Front view
    )
    
    # Update layout
    fig.update_layout(
        scene=dict(
            xaxis_title='X (AU)',
            yaxis_title='Y (AU)',
            zaxis_title='Z (AU)',
            aspectmode='cube',
            camera=camera
        ),
        height=700,  # Increased height
        title=dict(
            text=f"{title} - Asteroid {st.session_state.selected_asteroid}",
            y=0.95
        ),
        margin=dict(l=0, r=0, t=50, b=0)  # Reduced margins
    )
    
    return fig

def create_trajectory_difference_plot(diff_data, coord, plot_type="position"):
    """Create trajectory difference plot for either position or velocity"""
    fig = go.Figure()
    
    diff_data = diff_data.copy()
    diff_data['date'] = diff_data['time'].apply(julian_to_gregorian)
    
    y_col = f"{'v' if plot_type == 'velocity' else ''}{coord}_diff"
    
    fig.add_trace(go.Scatter(
        x=diff_data['date'],
        y=diff_data[y_col],
        mode='lines',
        name=f"{coord.upper()} {plot_type.capitalize()} Difference"
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=50, r=50, t=30, b=30),
        xaxis_title="Date",
        yaxis_title=f"Difference ({'AU/day' if plot_type == 'velocity' else 'AU'})",
        template="plotly_white",
        showlegend=True
    )
    
    return fig

def create_coordinate_comparison_plot(actual_data, model_data, coordinate):
    """Create comparison plot for a specific coordinate"""
    fig = go.Figure()
    
    # Convert time to dates for both datasets
    actual_data['date'] = actual_data['time'].apply(julian_to_gregorian)
    model_data['date'] = model_data['time'].apply(julian_to_gregorian)
    
    # Add actual trajectory
    fig.add_trace(go.Scatter(
        x=actual_data['date'],
        y=actual_data[coordinate],
        mode='lines',
        name='Actual',
        line=dict(color='blue', width=2)
    ))
    
    # Add model trajectory
    fig.add_trace(go.Scatter(
        x=model_data['date'],
        y=model_data[coordinate],
        mode='lines',
        name='Model',
        line=dict(color='magenta', width=2, dash='dash')
    ))
    
    fig.update_layout(
        height=250,
        xaxis_title="Date",
        yaxis_title=f"{coordinate.upper()} (AU)",
        showlegend=True,
        margin=dict(l=50, r=50, t=30, b=30),
        template="plotly_white"
    )
    
    return fig

def display_overview():
    st.header("Overview")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("### PINN Model")
        st.write("Physics-Informed Neural Network")
        st.markdown("**Key Equation:**")
        st.latex(r"\frac{d^2\mathbf{r}}{dt^2} = -\frac{GM\mathbf{r}}{|\mathbf{r}|^3} + \mathbf{F}_{other}")

    with col2:
        st.write("### N-body Model")
        st.write("Gravitational N-body Simulation")
        st.markdown("**Key Equation:**")
        st.latex(r"\frac{d^2\mathbf{r}_i}{dt^2} = \sum_{j \neq i} G m_j \frac{\mathbf{r}_j - \mathbf{r}_i}{|\mathbf{r}_j - \mathbf{r}_i|^3}")

    with col3:
        st.write("### Transformer Model")
        st.write("Deep Learning Sequence Model")
        
    st.write("### Dataset Overview")
    stats_col1, stats_col2, stats_col3 = st.columns(3)
    
    with stats_col1:
        st.metric("Total Asteroids", "9")
    with stats_col2:
        st.metric("Time Period", "2000-2024")
    with stats_col3:
        st.metric("Data Points", "1.2M+")

    st.write("### How to Use This Dashboard")
    st.write("""
    1. Select an asteroid from the dropdown menu
    2. Explore trajectory visualizations
    3. Compare model performances
    4. Analyze detailed metrics
    """)

def display_actual_trajectory():
    """Display the actual trajectory tab content"""
    st.header("Actual Trajectory")
    
    if not st.session_state.data_loaded:
        st.error("No data loaded. Please select an asteroid.")
        return
        
    actual_data = st.session_state.data.get('actual')
    if actual_data is None:
        st.error("No actual trajectory data available")
        return
    
    # Create single column for visualization
    col1, col2 = st.columns([4, 1])
    
    with col1:
        fig = display_trajectory_plot(actual_data, "Actual Trajectory", is_actual=True, camera_preset="top")
        st.plotly_chart(fig, use_container_width=True, key="actual_trajectory_plot")


def display_overview():
    """Display overview tab content"""
    st.header("Overview")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("### PINN Model")
        st.write("Physics-Informed Neural Network")
        st.markdown("**Key Equation:**")
        st.latex(r"\frac{d^2\mathbf{r}}{dt^2} = -\frac{GM\mathbf{r}}{|\mathbf{r}|^3} + \mathbf{F}_{other}")

    with col2:
        st.write("### N-body Model")
        st.write("Gravitational N-body Simulation")
        st.markdown("**Key Equation:**")
        st.latex(r"\frac{d^2\mathbf{r}_i}{dt^2} = \sum_{j \neq i} G m_j \frac{\mathbf{r}_j - \mathbf{r}_i}{|\mathbf{r}_j - \mathbf{r}_i|^3}")

    with col3:
        st.write("### Transformer Model")
        st.write("Deep Learning Sequence Model")
        
    st.write("### Dataset Overview")
    stats_col1, stats_col2, stats_col3 = st.columns(3)
    
    with stats_col1:
        st.metric("Total Asteroids", "9")
    with stats_col2:
        st.metric("Time Period", "2000-2024")
    with stats_col3:
        st.metric("Data Points", "1.2M+")

    st.write("### How to Use This Dashboard")
    st.write("""
    1. Select an asteroid from the dropdown menu
    2. Explore trajectory visualizations
    3. Compare model performances
    4. Analyze detailed metrics
    """)

def display_actual_trajectory():
    """Display the actual trajectory tab content"""
    st.header("Actual Trajectory")
    
    if not st.session_state.data_loaded:
        st.error("No data loaded. Please select an asteroid.")
        return
        
    actual_data = st.session_state.data.get('actual')
    if actual_data is None:
        st.error("No actual trajectory data available")
        return
    
    # Create single column for visualization
    col1, col2 = st.columns([4, 1])
    
    with col1:
        fig = display_trajectory_plot(actual_data, "Actual Trajectory", is_actual=True, camera_preset="top")
        st.plotly_chart(fig, use_container_width=True, key="actual_trajectory_plot")

def display_model_trajectories():
    """Display model trajectories tab content with unified model selection"""
    st.header("Model Trajectories")
    
    if not st.session_state.data_loaded:
        st.error("No data loaded. Please select an asteroid.")
        return

    # Model selection at the top
    model_options = ["PINN", "N-body", "Transformer"]
    selected_model = st.radio(
        "Select Model",
        model_options,
        horizontal=True,
        key="model_selector"
    )
    
    # Map model names to data keys
    model_key_map = {
        "PINN": "pinn",
        "N-body": "nbody",
        "Transformer": "transformer"
    }
    current_model_key = model_key_map[selected_model]
    
    # Main container
    main_container = st.container()
    
    with main_container:
        # First section: 3D Trajectory and Differences
        st.markdown("### 3D Trajectory and Differences")
        vis_col, diff_col = st.columns([1.2, 0.8])
        
        # 3D Trajectory Visualization
        with vis_col:
            model_data = st.session_state.data.get(current_model_key)
            if model_data is not None:
                fig = display_trajectory_plot(model_data, f"{selected_model} Model")
                st.plotly_chart(fig, use_container_width=True, key=f"model_trajectory_{selected_model}")
            else:
                st.error(f"No trajectory data available for {selected_model} model")
        
        # Trajectory Differences
        with diff_col:
            st.markdown("### Trajectory Differences")
            
            # Coordinate selector
            selected_coord = st.radio(
                "Select Coordinate",
                ["X", "Y", "Z"],
                horizontal=True,
                key="coordinate_selector"
            ).lower()
            
            # Get difference data for current model
            diff_data = st.session_state.data['differences'].get(current_model_key)
            
            if diff_data is not None:
                # Position differences plot
                fig_pos = create_trajectory_difference_plot(
                    diff_data, 
                    selected_coord,
                    "position"
                )
                st.plotly_chart(fig_pos, use_container_width=True, 
                              key=f"diff_pos_{selected_model}_{selected_coord}")
                
                # Velocity differences plot
                fig_vel = create_trajectory_difference_plot(
                    diff_data,
                    selected_coord,
                    "velocity"
                )
                st.plotly_chart(fig_vel, use_container_width=True, 
                              key=f"diff_vel_{selected_model}_{selected_coord}")
            else:
                st.error(f"No difference data available for {selected_model} model")
        
        # Model Performance Metrics
        st.markdown("### Model Performance Metrics")
        model_summary = load_analysis_summary(
            st.session_state.selected_asteroid,
            selected_model
        )
        
        if model_summary:
            rmse_col1, rmse_col2 = st.columns(2)
            with rmse_col1:
                st.metric("RMSE Position", format_number(model_summary.get('rmse_position', 'N/A')))
            with rmse_col2:
                st.metric("RMSE Velocity", format_number(model_summary.get('rmse_velocity', 'N/A')))
        
        # Coordinate Comparisons
        st.markdown("### Coordinate Comparisons")
        actual_data = st.session_state.data['actual']
        model_data = st.session_state.data.get(current_model_key)
        
        if actual_data is not None and model_data is not None:
            for coord in ['x', 'y', 'z']:
                st.markdown(f"#### {coord.upper()} Coordinate")
                fig = create_coordinate_comparison_plot(
                    actual_data,
                    model_data,
                    coord
                )
                st.plotly_chart(fig, use_container_width=True, 
                              key=f"coord_comparison_{selected_model}_{coord}")
def main():
    """Main function to run the dashboard"""
    # Initialize data
    asteroid_ids = [16, 21, 243, 253, 433, 4179, 25143, 101955, 162173]
    asteroid_info = load_asteroid_info()

    # Header Section
    st.write("""
    <div style='padding: 0.5rem 0; margin-bottom: 0.5rem'>
        <h1 style='margin: 0; padding: 0'>☄️ Asteroid Trajectory Dashboard</h1>
    </div>
    """, unsafe_allow_html=True)

    # Asteroid Selection and Info Display
    col1, col2, col3 = st.columns([2, 2, 2])
    with col1:
        selected_asteroid = st.selectbox(
            "Select Asteroid",
            asteroid_ids,
            index=asteroid_ids.index(st.session_state.get('selected_asteroid', asteroid_ids[0])),
            key="global_asteroid_select",
            label_visibility="collapsed"
        )
        st.session_state.selected_asteroid = selected_asteroid
        
    # Load data for selected asteroid
    st.session_state.data = lazy_load_data(st.session_state.selected_asteroid)

    if str(st.session_state.selected_asteroid) in asteroid_info:
        info = asteroid_info[str(st.session_state.selected_asteroid)]
        
        # Display asteroid information
        with col2:
            st.markdown(f"""
            <div class="asteroid-header">
                <strong>Name:</strong> {info['name']}<br>
                <strong>Type:</strong> {info['type']} | <strong>Group:</strong> {info['group']}
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div class="asteroid-header">
                <strong>Orbital Period:</strong> {info['orbital_period']} days<br>
                <strong>Mass:</strong> {format_mass(info['mass'])} | <strong>Diameter:</strong> {info['diameter']} km
            </div>
            """, unsafe_allow_html=True)

        # Metrics display
        metrics_cols = st.columns(3)
        with metrics_cols[0]:
            st.metric("Aphelion", f"{info['aphelion']} AU")
        with metrics_cols[1]:
            st.metric("Perihelion", f"{info['perihelion']} AU")
        with metrics_cols[2]:
            st.metric("Eccentricity", info['eccentricity'])

    # Create tabs
    tabs = st.tabs(["Overview", "Actual Trajectory", "Model Trajectories"])
    
    # Show content based on selected tab
    with tabs[0]:
        display_overview()
    with tabs[1]:
        display_actual_trajectory()
    with tabs[2]:
        display_model_trajectories()

    # Memory cleanup
    clean_memory()

if __name__ == "__main__":
    main()