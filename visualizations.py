import streamlit as st
import plotly.graph_objects as go
import numpy as np
from datetime import datetime
from utils import julian_to_gregorian, calculate_error_metrics
from config import MODEL_CONFIGS

# Define plot styling constants
PLOT_STYLE = {
    'background_color': '#FFFFFF',    # White background
    'grid_color': '#E0E0E0',         # Light gray grid
    'text_color': '#000000',         # Black text
    'line_width': 3,                 # Thick lines
    'colors': {
        'actual': '#0066FF',         # Bright blue
        'PINN': '#FF1493',          # Deep pink
        'N-body': '#00CC00',        # Bright green
        'Transformer': '#9933FF',    # Bright purple
        'LSTM': '#FF8C00',          # Dark orange
        'Sun': '#FFD700',           # Gold
        'Mercury': '#9370DB',        # Medium purple
        'Venus': '#FF6B6B',         # Coral pink
        'Earth': '#4169E1',         # Royal blue
        'Mars': '#FF4500',          # Orange red
        'Jupiter': '#CD853F'        # Peru brown
    }
}



def create_3d_trajectory_plot(data, title, color='#0066FF', show_planets=True):
    """Create a 3D trajectory plot with planets"""
    fig = go.Figure()
    
    # Add Sun
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0],
        mode='markers',
        marker=dict(
            size=15,
            color=PLOT_STYLE['colors']['Sun'],
            symbol='circle'
        ),
        name='Sun'
    ))
    
    # Add planets if enabled and available
    if show_planets and 'planets' in st.session_state.data:
        planet_list = ['Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter']
        for planet in planet_list:
            if (st.session_state.show_planets.get(planet, True) and 
                planet in st.session_state.data['planets']):
                planet_data = st.session_state.data['planets'][planet]
                if planet_data is not None:
                    fig.add_trace(go.Scatter3d(
                        x=planet_data['x'],
                        y=planet_data['y'],
                        z=planet_data['z'],
                        mode='lines',
                        name=f'{planet} Orbit',
                        line=dict(
                            color=PLOT_STYLE['colors'][planet],
                            width=PLOT_STYLE['line_width'] - 1
                        )
                    ))
    
    # Add main trajectory
    fig.add_trace(go.Scatter3d(
        x=data['x'],
        y=data['y'],
        z=data['z'],
        mode='lines',
        line=dict(
            color=color,
            width=PLOT_STYLE['line_width']
        ),
        name=title
    ))
    
    # Update layout with light theme
    fig.update_layout(
        scene=dict(
            camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=1.5, y=1.5, z=1.5)
            ),
            xaxis=dict(
                title='X (AU)',
                gridcolor=PLOT_STYLE['grid_color'],
                showbackground=True,
                backgroundcolor=PLOT_STYLE['background_color']
            ),
            yaxis=dict(
                title='Y (AU)',
                gridcolor=PLOT_STYLE['grid_color'],
                showbackground=True,
                backgroundcolor=PLOT_STYLE['background_color']
            ),
            zaxis=dict(
                title='Z (AU)',
                gridcolor=PLOT_STYLE['grid_color'],
                showbackground=True,
                backgroundcolor=PLOT_STYLE['background_color']
            ),
            aspectmode='cube'
        ),
        height=700,
        margin=dict(l=0, r=0, t=30, b=0),
        paper_bgcolor=PLOT_STYLE['background_color'],
        plot_bgcolor=PLOT_STYLE['background_color'],
        font=dict(color=PLOT_STYLE['text_color'])
    )
    
    return fig

def create_trajectory_difference_plot(diff_data, coord, plot_type="position"):
    """Create trajectory difference plot"""
    fig = go.Figure()
    
    # Convert Julian dates to Gregorian
    dates = [julian_to_gregorian(t) for t in diff_data['time']]
    
    # Select the appropriate column based on coordinate and type
    if plot_type == "position":
        y_col = f"{coord}_diff"
    else:  # velocity differences might not be available
        return None
    
    if y_col not in diff_data.columns:
        return None
    
    fig.add_trace(go.Scatter(
        x=dates,
        y=diff_data[y_col],
        mode='lines',
        name=f"{coord.upper()} {plot_type.capitalize()} Difference",
        line=dict(
            color=PLOT_STYLE['colors']['actual'],
            width=PLOT_STYLE['line_width']
        )
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=50, r=50, t=30, b=30),
        xaxis_title="Date",
        yaxis_title=f"Difference (AU)",
        paper_bgcolor=PLOT_STYLE['background_color'],
        plot_bgcolor=PLOT_STYLE['background_color'],
        font=dict(color=PLOT_STYLE['text_color']),
        xaxis=dict(
            gridcolor=PLOT_STYLE['grid_color'],
            showgrid=True
        ),
        yaxis=dict(
            gridcolor=PLOT_STYLE['grid_color'],
            showgrid=True
        )
    )
    
    return fig

def create_coordinate_comparison_plot(actual_data, model_data, coordinate):
    """Create comparison plot"""
    fig = go.Figure()
    
    # Convert Julian dates to Gregorian
    actual_dates = [julian_to_gregorian(t) for t in actual_data['time']]
    model_dates = [julian_to_gregorian(t) for t in model_data['time']]
    
    # Add actual trajectory
    fig.add_trace(go.Scatter(
        x=actual_dates,
        y=actual_data[coordinate],
        mode='lines',
        name='Actual',
        line=dict(
            color=PLOT_STYLE['colors']['actual'],
            width=PLOT_STYLE['line_width']
        )
    ))
    
    # Add model trajectory
    fig.add_trace(go.Scatter(
        x=model_dates,
        y=model_data[coordinate],
        mode='lines',
        name='Model',
        line=dict(
            color=PLOT_STYLE['colors'][st.session_state.current_model],
            width=PLOT_STYLE['line_width'],
            dash='dash'
        )
    ))
    
    fig.update_layout(
        height=300,
        xaxis_title="Date",
        yaxis_title=f"{coordinate.upper()} (AU)",
        showlegend=True,
        margin=dict(l=50, r=50, t=30, b=30),
        paper_bgcolor=PLOT_STYLE['background_color'],
        plot_bgcolor=PLOT_STYLE['background_color'],
        font=dict(color=PLOT_STYLE['text_color']),
        xaxis=dict(
            gridcolor=PLOT_STYLE['grid_color'],
            showgrid=True
        ),
        yaxis=dict(
            gridcolor=PLOT_STYLE['grid_color'],
            showgrid=True
        )
    )
    
    return fig

def display_model_trajectories():
    """Display model trajectories tab content"""
    st.header("Model Trajectories")
    
    if not st.session_state.data_loaded:
        st.warning("Please select an asteroid to view model predictions.")
        return
    
    # Model selection with radio buttons
    selected_model = st.radio(
        "Select Model",
        list(MODEL_CONFIGS.keys()),
        horizontal=True,
        key='model_selector'
    )
    
    # Update current model in session state
    st.session_state.current_model = selected_model
    
    model_key = selected_model.lower()
    model_data = st.session_state.data.get(model_key)
    actual_data = st.session_state.data.get('actual')
    diff_data = st.session_state.data.get('differences', {}).get(model_key)
    
    if model_data is None or actual_data is None:
        st.error("Model or actual data not available")
        return
    
    # Create main sections
    st.markdown("### 3D Trajectory and Differences")
    vis_col, diff_col = st.columns([1.2, 0.8])
    
    with vis_col:
        # 3D trajectory plot
        fig_3d = create_3d_trajectory_plot(
            model_data,
            f"{selected_model} Prediction",
            color=PLOT_STYLE['colors'][selected_model]
        )
        st.plotly_chart(fig_3d, use_container_width=True)
        
        # Planet visibility toggles
        st.write("### Planet Visibility")
        planet_cols = st.columns(5)
        for idx, planet in enumerate(['Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter']):
            with planet_cols[idx]:
                st.session_state.show_planets[planet] = st.checkbox(
                    planet,
                    value=st.session_state.show_planets.get(planet, True),
                    key=f"model_planet_visibility_{planet}"
                )
    
    with diff_col:
        # Trajectory differences section
        st.markdown("### Trajectory Differences")
        
        # Coordinate selector
        selected_coord = st.radio(
            "Select Coordinate",
            ["X", "Y", "Z"],
            horizontal=True,
            key="coordinate_selector"
        ).lower()
        
        if diff_data is not None:
            # Position differences plot
            fig_pos = create_trajectory_difference_plot(
                diff_data,
                selected_coord,
                "position"
            )
            if fig_pos is not None:
                st.plotly_chart(fig_pos, use_container_width=True)
    
    # Model Performance Metrics
    st.markdown("### Model Performance Metrics")
    metrics = calculate_error_metrics(actual_data[['x', 'y', 'z']], model_data[['x', 'y', 'z']])
    
    if metrics:
        metric_cols = st.columns(4)
        with metric_cols[0]:
            st.metric("Position RMSE (AU)", f"{metrics['rmse']:.2e}")
        with metric_cols[1]:
            st.metric("Position MAE (AU)", f"{metrics['mae']:.2e}")
        with metric_cols[2]:
            st.metric("Max Error (AU)", f"{metrics['max_error']:.2e}")
        with metric_cols[3]:
            st.metric("Prediction Time", "N/A")
    
    # Coordinate Comparisons
    st.markdown("### Coordinate Comparisons")
    for coord in ['x', 'y', 'z']:
        st.markdown(f"#### {coord.upper()} Coordinate")
        fig = create_coordinate_comparison_plot(
            actual_data,
            model_data,
            coord
        )
        st.plotly_chart(fig, use_container_width=True)

def display_actual_trajectory():
    """Display actual trajectory tab content"""
    st.header("Actual Trajectory")
    
    if not st.session_state.data_loaded:
        st.warning("Please select an asteroid to view trajectory data.")
        return
    
    actual_data = st.session_state.data.get('actual')
    if actual_data is None:
        st.error("No actual trajectory data available")
        return
    
    # Create visualization columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # 3D trajectory plot with planets
        fig_3d = create_3d_trajectory_plot(
            actual_data,
            "Actual Trajectory",
            color=PLOT_STYLE['colors']['actual'],
            show_planets=True
        )
        st.plotly_chart(fig_3d, use_container_width=True)
        
        # Planet visibility toggles
        st.write("### Planet Visibility")
        planet_cols = st.columns(5)
        for idx, planet in enumerate(['Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter']):
            with planet_cols[idx]:
                st.session_state.show_planets[planet] = st.checkbox(
                    planet,
                    value=st.session_state.show_planets.get(planet, True),
                    key=f"actual_planet_visibility_{planet}"
                )
    
    with col2:
        # Convert time to Gregorian dates
        dates = [julian_to_gregorian(t) for t in actual_data['time']]
        
        # Coordinate plots
        for coord in ['x', 'y', 'z']:
            st.write(f"#### {coord.upper()} Coordinate")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=dates,
                y=actual_data[coord],
                mode='lines',
                name=f"{coord.upper()} Position",
                line=dict(
                    color=PLOT_STYLE['colors']['actual'],
                    width=PLOT_STYLE['line_width']
                )
            ))
            fig.update_layout(
                xaxis_title="Date",
                yaxis_title=f"{coord.upper()} (AU)",
                height=300,
                margin=dict(l=50, r=20, t=30, b=30),
                paper_bgcolor=PLOT_STYLE['background_color'],
                plot_bgcolor=PLOT_STYLE['background_color'],
                font=dict(color=PLOT_STYLE['text_color']),
                xaxis=dict(gridcolor=PLOT_STYLE['grid_color']),
                yaxis=dict(gridcolor=PLOT_STYLE['grid_color'])
            )
            st.plotly_chart(fig, use_container_width=True)

def display_overview():
    """Display overview tab content"""
    st.header("Overview")
    
    # Introduction
    st.markdown("""
    This dashboard visualizes asteroid trajectories using different prediction models. 
    Each model represents a different approach to predicting asteroid movement through space.
    """)
    
    # Model descriptions
    st.subheader("Model Architectures")
    
    model_cols = st.columns(4)
    
    with model_cols[0]:
        st.markdown("### PINN Model")
        st.markdown("""
        **Physics-Informed Neural Network**
        - Combines neural networks with physical laws
        - Incorporates gravitational constraints
        - Balances data-driven and physics-based learning
        """)
        st.latex(r"\frac{d^2\mathbf{r}}{dt^2} = -\frac{GM\mathbf{r}}{|\mathbf{r}|^3} + \mathbf{F}_{other}")
    
    with model_cols[1]:
        st.markdown("### N-body Model")
        st.markdown("""
        **Gravitational N-body Simulation**
        - Full gravitational interaction simulation
        - Accounts for all planetary influences
        - High physical accuracy
        """)
        st.latex(r"\frac{d^2\mathbf{r}_i}{dt^2} = \sum_{j \neq i} G m_j \frac{\mathbf{r}_j - \mathbf{r}_i}{|\mathbf{r}_j - \mathbf{r}_i|^3}")
    
    with model_cols[2]:
        st.markdown("### Transformer Model")
        st.markdown("""
        **Deep Learning Sequence Model**
        - Multi-head attention mechanism
        - Captures long-range dependencies
        - Position encodings for temporal data
        - Key Features:
            - Self-attention layers
            - Feed-forward networks
            - Layer normalization
        """)
    
    with model_cols[3]:
        st.markdown("### LSTM Model")
        st.markdown("""
        **Long Short-Term Memory Network**
        - Sequential learning architecture
        - Memory cells for long-term patterns
        - Key Components:
            - Forget gate
            - Input gate
            - Output gate
            - Cell state
        """)
    
    # Dataset Overview
    st.markdown("---")
    st.subheader("Dataset Overview")
    
    # Dataset statistics in metrics
    stat_cols = st.columns(4)
    with stat_cols[0]:
        st.metric("Total Asteroids", "9")
    with stat_cols[1]:
        st.metric("Time Period", "2000-2024")
    with stat_cols[2]:
        st.metric("Total Data Points", "1.2M+")
    with stat_cols[3]:
        st.metric("Prediction Window", "5 Years")
    
    # Feature description
    st.markdown("### Features")
    feature_cols = st.columns(2)
    
    with feature_cols[0]:
        st.markdown("""
        **Trajectory Data**
        - Position (x, y, z) coordinates
        - Velocity components
        - Time series format
        - High-precision measurements
        """)
    
    with feature_cols[1]:
        st.markdown("""
        **Additional Information**
        - Planetary positions
        - Solar system dynamics
        - Gravitational influences
        - Orbital parameters
        """)
    
    # Usage Instructions
    st.markdown("### How to Use This Dashboard")
    st.markdown("""
    1. **Select an Asteroid**: Choose from the dropdown menu at the top
    2. **View Actual Trajectory**: See the real path in the 'Actual Trajectory' tab
    3. **Compare Models**: In the 'Model Trajectories' tab:
        - Select different models using radio buttons
        - Compare prediction accuracy
        - View error metrics
        - Analyze coordinate-wise differences
    4. **Customize View**:
        - Toggle planet visibility
        - Select different coordinates
        - Interact with 3D visualizations
    """)
    
    # Notes and Explanations
    st.markdown("### Important Notes")
    st.markdown("""
    - All distances are in Astronomical Units (AU)
    - Times are shown in Gregorian dates
    - Trajectory differences show deviation from actual path
    - RMSE (Root Mean Square Error) indicates prediction accuracy
    - Planet orbits can be toggled for clearer visualization
    """)
    
    # Footer
    st.markdown("---")
    st.caption("For more information about the models and methodology, please refer to the documentation.")