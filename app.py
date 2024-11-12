import streamlit as st
import logging
from datetime import datetime
from config import initialize_config
from data_loader import load_asteroid_info, lazy_load_data
from visualizations import display_overview, display_actual_trajectory, display_model_trajectories
from utils import clean_memory, format_mass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def initialize_session_state():
    """Initialize all session state variables"""
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
    if 'current_model' not in st.session_state:
        st.session_state.current_model = 'PINN'
    if 'current_coordinate' not in st.session_state:
        st.session_state.current_coordinate = 'x'

def main():
    """Main function to run the dashboard"""
    # Initialize configurations
    initialize_config()
    
    # Header
    st.title("☄️ Asteroid Trajectory Dashboard")
    
    # Available asteroid IDs
    asteroid_ids = [16, 21, 243, 253, 433, 4179, 25143, 101955, 162173]
    
    # Asteroid selection
    col1, col2, col3 = st.columns([2, 2, 2])
    
    with col1:
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
    with st.spinner("Loading data..."):
        asteroid_info = load_asteroid_info()
        data = lazy_load_data(selected_asteroid)
    
    # Display asteroid information
    if str(selected_asteroid) in asteroid_info:
        info = asteroid_info[str(selected_asteroid)]
        
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

        # Display metrics
        metric_cols = st.columns(3)
        with metric_cols[0]:
            st.metric("Aphelion", f"{info['aphelion']} AU")
        with metric_cols[1]:
            st.metric("Perihelion", f"{info['perihelion']} AU")
        with metric_cols[2]:
            st.metric("Eccentricity", info['eccentricity'])
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Overview", "Actual Trajectory", "Model Trajectories"])
    
    with tab1:
        display_overview()
    with tab2:
        display_actual_trajectory()
    with tab3:
        display_model_trajectories()
    
    # Footer
    st.markdown("---")
    footer_col1, footer_col2 = st.columns(2)
    with footer_col1:
        st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    with footer_col2:
        st.caption("Dashboard Version: 1.0.0")
    
    # Clean up
    clean_memory()

if __name__ == "__main__":
    try:
        initialize_session_state()
        main()
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        logger.exception("Application error")