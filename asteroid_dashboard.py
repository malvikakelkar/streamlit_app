# Import required libraries
import streamlit as st
import plotly.graph_objs as go
import pandas as pd
import json
import numpy as np
import plotly.express as px
from sklearn.metrics import mean_squared_error
from pathlib import Path
from datetime import datetime, timedelta
from plotly.subplots import make_subplots
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle
from io import BytesIO
import logging 
import gc

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

# Memory management
def clean_memory():
    gc.collect()

# Utility functions
def julian_to_gregorian(julian_date):
    return datetime.fromordinal(int(julian_date - 1721425.5)) + timedelta(days=julian_date % 1)

def format_number(num):
    try:
        return f"{float(num):.5e}"
    except (ValueError, TypeError):
        return "N/A"

def format_mass(mass_str):
    try:
        mass = float(mass_str)
        return f"{mass:.2e} kg"
    except ValueError:
        return mass_str
    
def create_model_comparison_pdf(asteroid_ids):
    """
    Creates a PDF comparing model performances for different asteroids.
    """
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    elements = []

    # Create table data
    data = [['Asteroid ID', 'PINN RMSE', 'N-body RMSE', 'Transformer RMSE']]
    
    for asteroid_id in asteroid_ids:
        pinn_summary = load_analysis_summary(asteroid_id, "PINN")
        nbody_summary = load_analysis_summary(asteroid_id, "N-body")
        transformer_summary = load_analysis_summary(asteroid_id, "Transformer")
        
        if all([pinn_summary, nbody_summary, transformer_summary]):
            data.append([
                str(asteroid_id),
                f"Pos: {format_number(pinn_summary['rmse_position'])}\n"
                f"Vel: {format_number(pinn_summary['rmse_velocity'])}",
                f"Pos: {format_number(nbody_summary['rmse_position'])}\n"
                f"Vel: {format_number(nbody_summary['rmse_velocity'])}",
                f"Pos: {format_number(transformer_summary['rmse_position'])}\n"
                f"Vel: {format_number(transformer_summary['rmse_velocity'])}"
            ])

    # Create table and set style
    table = Table(data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
        ('TEXTCOLOR', (0,0), (-1,0), colors.black),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,0), 14),
        ('BOTTOMPADDING', (0,0), (-1,0), 12),
        ('BACKGROUND', (0,1), (-1,-1), colors.white),
        ('TEXTCOLOR', (0,1), (-1,-1), colors.black),
        ('ALIGN', (0,1), (-1,-1), 'CENTER'),
        ('FONTNAME', (0,1), (-1,-1), 'Helvetica'),
        ('FONTSIZE', (0,1), (-1,-1), 12),
        ('TOPPADDING', (0,1), (-1,-1), 6),
        ('BOTTOMPADDING', (0,1), (-1,-1), 6),
        ('GRID', (0,0), (-1,-1), 1, colors.black)
    ]))

    elements.append(table)
    
    # Build PDF document
    doc.build(elements)
    buffer.seek(0)
    return buffer

# Data loading functions with caching
@st.cache_data(ttl=3600)
def load_asteroid_info():
    logger.info("Loading asteroid info...")
    try:
        with open('asteroid_info.json', 'r') as f:
            data = json.load(f)
            return data
    except FileNotFoundError:
        logger.error("asteroid_info.json file not found")
        return {}
    except Exception as e:
        logger.error(f"Error loading asteroid info: {str(e)}")
        return {}

@st.cache_data(ttl=3600)
def load_trajectory_data(asteroid_id, data_type):
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
    try:
        path = Path(f'actual_data/asteroid_{asteroid_id}/{planet}_data.csv')
        return pd.read_csv(path)
    except Exception as e:
        logger.error(f"Error loading planet data: {str(e)}")
        return None

@st.cache_data(ttl=3600)
def load_analysis_summary(asteroid_id, model):
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

# Lazy loading implementation
def lazy_load_data(asteroid_id):
    if not st.session_state.data_loaded or st.session_state.data.get('asteroid_id') != asteroid_id:
        st.session_state.data = {
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
        st.session_state.data_loaded = True
    return st.session_state.data


# Plot creation functions
@st.cache_data(ttl=3600)
def create_cached_plot(plot_type, data, title, **kwargs):
    if plot_type == '3d':
        return create_3d_plot(data['trajectory'], data['planets'], title)
    elif plot_type == '2d':
        return create_2d_comparison_plot(data['actual'], data['model'], title, kwargs.get('y_column'))
    elif plot_type == 'difference':
        return create_trajectory_difference_plot(data, title)
    return None

def create_3d_plot(asteroid_data, planet_data, title):
    fig = go.Figure()

    fig.add_trace(go.Scatter3d(x=[0], y=[0], z=[0], mode='markers', 
                               marker=dict(size=10, color='yellow'), name='Sun'))

    fig.add_trace(go.Scatter3d(x=asteroid_data['x'], y=asteroid_data['y'], z=asteroid_data['z'],
                               mode='lines', name='Asteroid', line=dict(color='darkblue', width=4)))

    colors = {'Mercury': 'purple', 'Venus': 'orange', 'Earth': 'blue', 'Mars': 'red', 'Jupiter': 'brown'}
    for planet, color in colors.items():
        planet_df = planet_data[planet]
        fig.add_trace(go.Scatter3d(x=planet_df['x'], y=planet_df['y'], z=planet_df['z'],
                                   mode='lines', name=planet, line=dict(color=color, width=2)))

    fig.update_layout(
        title=dict(text=title, font=dict(size=24)),
        scene=dict(
            xaxis_title='X (AU)', 
            yaxis_title='Y (AU)', 
            zaxis_title='Z (AU)',
            aspectmode='cube'
        ),
        template='plotly',
        height=800,
        width=1000,
        legend=dict(font=dict(size=16)),
        hovermode="closest"
    )
    return fig

def create_2d_comparison_plot(actual_data, model_data, title, y_column):
    fig = go.Figure()
    
    actual_data['date'] = pd.to_datetime(actual_data['time'].apply(julian_to_gregorian))
    model_data['date'] = pd.to_datetime(model_data['time'].apply(julian_to_gregorian))
    
    fig.add_trace(go.Scatter(x=actual_data['date'], y=actual_data[y_column],
                            mode='lines', name='Actual', 
                            line=dict(color='darkblue', width=3)))
    fig.add_trace(go.Scatter(x=model_data['date'], y=model_data[y_column],
                            mode='lines', name='Model', 
                            line=dict(color='magenta', width=3, dash='dash')))
    
    y_label = "Position (AU)" if y_column in ['x', 'y', 'z'] else "Velocity (AU/day)"
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=24)),
        xaxis_title='Date',
        yaxis_title=y_label,
        legend=dict(font=dict(size=16), orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=500,
        width=800,
        template="plotly",
        hovermode="x unified"
    )
    
    fig.update_xaxes(tickformat="%Y-%m", dtick="M3")
    return fig


def create_trajectory_difference_plot(diff_data, title):
    if diff_data is None or diff_data.empty:
        st.error(f"No data available for {title}")
        return None

    diff_cols = ['x_diff', 'y_diff', 'z_diff', 'vx_diff', 'vy_diff', 'vz_diff']
    available_cols = [col for col in diff_cols if col in diff_data.columns]
    
    if len(available_cols) == 0:
        st.error(f"No difference columns found in the data for {title}")
        return None

    num_cols = len(available_cols)
    rows = (num_cols + 2) // 3  # Round up to the nearest multiple of 3
    cols = min(3, num_cols)

    fig = make_subplots(rows=rows, cols=cols, subplot_titles=available_cols)
    
    # Convert Julian dates to datetime
    diff_data = diff_data.copy()  # Create a copy to avoid modifying original data
    diff_data['date'] = diff_data['time'].apply(julian_to_gregorian)
    
    colors = ['darkred', 'darkgreen', 'darkblue', 'darkorange', 'purple', 'teal']
    
    for i, col in enumerate(available_cols):
        row, col_num = i // 3 + 1, i % 3 + 1
        fig.add_trace(go.Scatter(x=diff_data['date'], y=diff_data[col], mode='lines', 
                                 name=col, line=dict(color=colors[i % len(colors)])), 
                      row=row, col=col_num)
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=24)),
        height=400 * rows,
        width=1200,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly",
        hovermode="x unified"
    )
    fig.update_xaxes(title_text='Date', tickformat="%Y-%m", dtick="M3")
    fig.update_yaxes(title_text='Difference')
    
    for i in range(num_cols):
        fig.update_yaxes(autorange=True, row=i//3+1, col=i%3+1)
    
    return fig


def display_model_analysis(col, model_name, summary):
    with col:
        with st.container():
            st.subheader(f"{model_name} Model Analysis")
            st.write(f"RMSE Position: {format_number(summary.get('rmse_position', 'N/A'))}")
            st.write(f"RMSE Velocity: {format_number(summary.get('rmse_velocity', 'N/A'))}")
            
            st.write("Deviations")
            deviations_data = {
                '': ['Min Position', 'Min Velocity', 'Max Position', 'Max Velocity'],
                'X': [
                    format_number(summary['minimum_deviations']['position'].get('x_diff', 'N/A')),
                    format_number(summary['minimum_deviations']['velocity'].get('vx_diff', 'N/A')),
                    format_number(summary['maximum_deviations']['position'].get('x_diff', 'N/A')),
                    format_number(summary['maximum_deviations']['velocity'].get('vx_diff', 'N/A'))
                ],
                'Y': [
                    format_number(summary['minimum_deviations']['position'].get('y_diff', 'N/A')),
                    format_number(summary['minimum_deviations']['velocity'].get('y_diff', 'N/A')),
                    format_number(summary['maximum_deviations']['position'].get('y_diff', 'N/A')),
                    format_number(summary['maximum_deviations']['velocity'].get('vy_diff', 'N/A'))
                ],
                'Z': [
                    format_number(summary['minimum_deviations']['position'].get('z_diff', 'N/A')),
                    format_number(summary['minimum_deviations']['velocity'].get('vz_diff', 'N/A')),
                    format_number(summary['maximum_deviations']['position'].get('z_diff', 'N/A')),
                    format_number(summary['maximum_deviations']['velocity'].get('vz_diff', 'N/A'))
                ]
            }
            deviations_df = pd.DataFrame(deviations_data)
            st.dataframe(deviations_df.set_index(''), hide_index=False)

            st.write("Maximum Absolute Differences")
            max_diff_data = {
                '': ['Position', 'Velocity'],
                'X': [
                    format_number(summary['max_absolute_differences'].get('x_diff', 'N/A')),
                    format_number(summary['max_absolute_differences'].get('vx_diff', 'N/A'))
                ],
                'Y': [
                    format_number(summary['max_absolute_differences'].get('y_diff', 'N/A')),
                    format_number(summary['max_absolute_differences'].get('vy_diff', 'N/A'))
                ],
                'Z': [
                    format_number(summary['max_absolute_differences'].get('z_diff', 'N/A')),
                    format_number(summary['max_absolute_differences'].get('vzw_diff', 'N/A'))
                ]
            }
            max_diff_df = pd.DataFrame(max_diff_data)
            st.dataframe(max_diff_df.set_index(''), hide_index=False)

# CSS Styling
st.markdown("""
<style>
    .block-container {
        padding-top: 2rem !important;  /* Increased from 1rem */
        padding-bottom: 3rem;
        padding-left: 5rem;
        padding-right: 5rem;
        margin-top: 1rem;  /* Added margin-top */
    }
    
    /* Add specific styling for the header section */
    .stApp header {
        padding-top: 1rem;
        margin-bottom: 1rem;
    }
    
    /* Adjust the container holding the asteroid info */
    div[data-testid="stVerticalBlock"] > div:first-child {
        padding-top: 1rem !important;
        margin-top: 1rem !important;
    }
    
    /* Your existing styles... */
    .stMarkdown {
        margin-bottom: 0;
    }
    .stSelectbox {
        margin-bottom: 0.5rem;
    }
    [data-testid="stMetricValue"] {
        margin-bottom: 0.5rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        margin-bottom: 0.5rem;
    }
    h1 {
        margin-top: 0.5rem !important;  /* Added small margin-top */
        margin-bottom: 1rem !important;
    }
    .streamlit-expanderHeader {
        padding: 1rem !important;
        margin-top: 1rem !important;
    }
    .streamlit-expanderContent {
        padding: 1rem !important;
    }
    [data-testid="stExpander"] {
        margin-bottom: 2rem;
    }
    [data-testid="metric-container"] {
        font-size: 120% !important;
    }
    .asteroid-info {
        font-size: 120%;
        line-height: 1.5;
        margin: 0.5rem 0;
    }
    [data-testid="stMetricValue"] {
        font-size: 140% !important;
        font-weight: 600 !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 110% !important;
    }
    .asteroid-header {
        font-size: 130%;
        font-weight: 500;
        margin-bottom: 0.5rem;
        padding-top: 0.5rem;  /* Added padding-top */
    }
    h1 {
        font-size: 2.5rem !important;
    }
    .stMarkdown div {
        margin-bottom: 0.5rem;
    }
    [data-testid="stMetricValue"] div {
        margin-bottom: 0.25rem !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize data
asteroid_ids = [16, 21, 243, 253, 433, 4179, 25143, 101955, 162173]
asteroid_info = load_asteroid_info()

if 'selected_asteroid' not in st.session_state:
    st.session_state.selected_asteroid = asteroid_ids[0]

# Memory cleanup on asteroid change
if st.session_state.get('last_asteroid') != st.session_state.selected_asteroid:
    clean_memory()
    st.session_state.last_asteroid = st.session_state.selected_asteroid

# Load data for selected asteroid
data = lazy_load_data(st.session_state.selected_asteroid)

# Header section
st.write("""
<div style='padding: 0.5rem 0; margin-bottom: 0.5rem'>
    <div style='display: flex; gap: 1rem; align-items: center'>
        <h1 style='margin: 0; padding: 0'>☄️ Asteroid Trajectory Dashboard</h1>
        <div style='flex-grow: 1'>
""", unsafe_allow_html=True)

# Asteroid selection and info display
col1, col2, col3 = st.columns([2, 2, 2])
with col1:
    st.session_state.selected_asteroid = st.selectbox(
        "Select Asteroid",
        asteroid_ids,
        index=asteroid_ids.index(st.session_state.selected_asteroid),
        key="global_asteroid_select",
        label_visibility="collapsed"
    )

if str(st.session_state.selected_asteroid) in asteroid_info:
    info = asteroid_info[str(st.session_state.selected_asteroid)]
    
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

st.write("</div></div></div>", unsafe_allow_html=True)


# Navigation tabs
tab_overview, tab_visuals, tab_comparisons, tab_models = st.tabs([
    "Overview", "Trajectory Visuals", "Trajectory Comparisons", "Model Comparisons"
])

# Overview Tab
with tab_overview:
    st.header("Overview")
    col_intro = st.container()
    with col_intro:
        st.markdown("""

        this section will be modified later

        """)

# Trajectory Visuals Tab
with tab_visuals:
    st.header("Trajectory Visuals")
    st.markdown("*Double-click any legend item to isolate that trajectory, or hide others*")

    # Get data from session state
    data = st.session_state.data
    actual_data = data['actual']
    pinn_data = data['pinn']
    nbody_data = data['nbody']
    transformer_data = data['transformer']
    planet_data = data['planets']

    visual_tabs = st.tabs(["Actual Trajectory", "PINN Model", "N-body Model", "Transformer Model"])

    for tab, data, model_name in zip(
        visual_tabs, 
        [actual_data, pinn_data, nbody_data, transformer_data],
        ["Actual", "PINN", "N-body", "Transformer"]
    ):
        with tab:
            fig = create_3d_plot(
                data, 
                planet_data, 
                f"{model_name} Trajectory of Asteroid {st.session_state.selected_asteroid}"
            )
            st.plotly_chart(fig, use_container_width=True)

# Trajectory Comparisons Tab
with tab_comparisons:
    st.header("Trajectory Comparisons")
    st.markdown("*Double-click any legend item to isolate that trajectory, or hide others*")

    # Get data from session state
    data = st.session_state.data
    actual_data = data['actual']
    pinn_data = data['pinn']
    nbody_data = data['nbody']
    transformer_data = data['transformer']
    differences = data['differences']

    comparison_tabs = st.tabs(["Position Comparisons", "Velocity Comparisons", "Trajectory Differences"])

    # Position Comparisons
    with comparison_tabs[0]:
        st.subheader("Position Comparisons")
        for coord in ['x', 'y', 'z']:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.plotly_chart(
                    create_2d_comparison_plot(actual_data, pinn_data, f"PINN vs Actual - {coord.upper()}", coord),
                    use_container_width=True,
                    key=f"pos_pinn_{coord}"
                )
            with col2:
                st.plotly_chart(
                    create_2d_comparison_plot(actual_data, nbody_data, f"N-body vs Actual - {coord.upper()}", coord),
                    use_container_width=True,
                    key=f"pos_nbody_{coord}"
                )
            with col3:
                st.plotly_chart(
                    create_2d_comparison_plot(actual_data, transformer_data, f"Transformer vs Actual - {coord.upper()}", coord),
                    use_container_width=True,
                    key=f"pos_transformer_{coord}"
                )

    # Velocity Comparisons
    with comparison_tabs[1]:
        st.subheader("Velocity Comparisons")
        for coord in ['vx', 'vy', 'vz']:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.plotly_chart(
                    create_2d_comparison_plot(actual_data, pinn_data, f"PINN vs Actual - {coord.upper()}", coord),
                    use_container_width=True,
                    key=f"vel_pinn_{coord}"
                )
            with col2:
                st.plotly_chart(
                    create_2d_comparison_plot(actual_data, nbody_data, f"N-body vs Actual - {coord.upper()}", coord),
                    use_container_width=True,
                    key=f"vel_nbody_{coord}"
                )
            with col3:
                st.plotly_chart(
                    create_2d_comparison_plot(actual_data, transformer_data, f"Transformer vs Actual - {coord.upper()}", coord),
                    use_container_width=True,
                    key=f"vel_transformer_{coord}"
                )

    # Trajectory Differences
    with comparison_tabs[2]:
        st.subheader("Trajectory Differences")
        
        for model, diff_data in [
            ("PINN", differences['pinn']),
            ("N-body", differences['nbody']),
            ("Transformer", differences['transformer'])
        ]:
            if diff_data is not None and not diff_data.empty:
                fig = create_trajectory_difference_plot(
                    diff_data,
                    f"{model} Trajectory Differences - Asteroid {st.session_state.selected_asteroid}"
                )
                if fig is not None:
                    st.plotly_chart(fig, key=f"diff_{model}")
            else:
                st.error(f"No difference data available for {model} model and Asteroid {st.session_state.selected_asteroid}")


# Model Comparisons Tab
with tab_models:
    st.header("Model Comparisons")

    # Create PDF download button
    pdf_buffer = create_model_comparison_pdf(asteroid_ids)
    st.download_button(
        label="Download Model Comparison PDF",
        data=pdf_buffer,
        file_name="model_comparison.pdf",
        mime="application/pdf"
    )

    # Load analysis summaries for each model
    pinn_summary = load_analysis_summary(st.session_state.selected_asteroid, "PINN")
    nbody_summary = load_analysis_summary(st.session_state.selected_asteroid, "N-body")
    transformer_summary = load_analysis_summary(st.session_state.selected_asteroid, "Transformer")

    # Display model analysis in columns
    col1, col2, col3 = st.columns(3)

    if pinn_summary:
        display_model_analysis(col1, "PINN", pinn_summary)
    if nbody_summary:
        display_model_analysis(col2, "N-body", nbody_summary)
    if transformer_summary:
        display_model_analysis(col3, "Transformer", transformer_summary)

    # RMSE Comparison Section
    if all([pinn_summary, nbody_summary, transformer_summary]):
        st.subheader("RMSE Comparison")
        
        metric_option = st.radio(
            "Select RMSE Metric",
            ["Position", "Velocity"],
            horizontal=True,
            key="rmse_metric_selector"
        )
        
        if metric_option == "Position":
            rmse_data = pd.DataFrame({
                'Model': ['PINN', 'N-body', 'Transformer'],
                'RMSE Position': [
                    float(pinn_summary['rmse_position']),
                    float(nbody_summary['rmse_position']),
                    float(transformer_summary['rmse_position'])
                ]
            })
            
            fig_position = go.Figure(data=[
                go.Bar(
                    x=rmse_data['Model'],
                    y=rmse_data['RMSE Position'],
                    text=rmse_data['RMSE Position'].apply(lambda x: f'{x:.2e}'),
                    textposition='outside',
                    marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1'],
                    hovertemplate='Model: %{x}<br>RMSE: %{y:.2e}<extra></extra>'
                )
            ])
            
            fig_position.update_layout(
                title=dict(
                    text='RMSE Position Comparison',
                    x=0.5,
                    xanchor='center',
                    font=dict(size=24)
                ),
                xaxis_title="Model",
                yaxis_title="RMSE Position (AU)",
                showlegend=False,
                height=500,
                template="plotly_white",
                yaxis=dict(
                    type='log',
                    dtick=0.30103,
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='LightGray'
                ),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            
            fig_position.update_xaxes(showgrid=False)
            st.plotly_chart(fig_position, use_container_width=True)
            
        else:  # Velocity RMSE
            rmse_data = pd.DataFrame({
                'Model': ['PINN', 'N-body', 'Transformer'],
                'RMSE Velocity': [
                    float(pinn_summary['rmse_velocity']),
                    float(nbody_summary['rmse_velocity']),
                    float(transformer_summary['rmse_velocity'])
                ]
            })
            
            fig_velocity = go.Figure(data=[
                go.Bar(
                    x=rmse_data['Model'],
                    y=rmse_data['RMSE Velocity'],
                    text=rmse_data['RMSE Velocity'].apply(lambda x: f'{x:.2e}'),
                    textposition='outside',
                    marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1'],
                    hovertemplate='Model: %{x}<br>RMSE: %{y:.2e}<extra></extra>'
                )
            ])
            
            fig_velocity.update_layout(
                title=dict(
                    text='RMSE Velocity Comparison',
                    x=0.5,
                    xanchor='center',
                    font=dict(size=24)
                ),
                xaxis_title="Model",
                yaxis_title="RMSE Velocity (AU/day)",
                showlegend=False,
                height=500,
                template="plotly_white",
                yaxis=dict(
                    type='log',
                    dtick=0.30103,
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='LightGray'
                ),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            
            fig_velocity.update_xaxes(showgrid=False)
            st.plotly_chart(fig_velocity, use_container_width=True)

        # RMSE Explanation
        with st.expander("Understanding RMSE Values"):
            if metric_option == "Position":
                st.markdown("""
                **Position RMSE Interpretation:**
                - Lower values indicate better position predictions
                - Values are in Astronomical Units (AU)
                - Logarithmic scale is used for better visualization of differences
                """)
            else:
                st.markdown("""
                **Velocity RMSE Interpretation:**
                - Lower values indicate better velocity predictions
                - Values are in AU per day
                - Logarithmic scale is used for better visualization of differences
                """)

    # Time Series Comparison
    st.subheader("Model Comparisons Over Time")
    st.markdown("*Double-click any legend item to isolate that model, or hide others*")

    coord = st.selectbox("Select coordinate", ['x', 'y', 'z'])
    
    fig = go.Figure()
    
    data = st.session_state.data
    if all(k in data for k in ['actual', 'pinn', 'nbody', 'transformer']):
        fig.add_trace(go.Scatter(x=data['actual']['time'], y=data['actual'][coord], 
                               mode='lines', name='Actual'))
        fig.add_trace(go.Scatter(x=data['pinn']['time'], y=data['pinn'][coord], 
                               mode='lines', name='PINN'))
        fig.add_trace(go.Scatter(x=data['nbody']['time'], y=data['nbody'][coord], 
                               mode='lines', name='N-body'))
        fig.add_trace(go.Scatter(x=data['transformer']['time'], y=data['transformer'][coord], 
                               mode='lines', name='Transformer'))

        fig.update_layout(
            title=f"{coord.upper()} Coordinate Comparison",
            xaxis_title="Time (UTC)",
            yaxis_title=f"{coord.upper()} (AU)",
            xaxis=dict(
                tickformat="%Y-%m-%d\n%H:%M:%S",
                tickangle=45,
            ),
            legend_title="Model",
            height=600,
            hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True)

# Help Section
with st.expander("Help & Information"):
    st.markdown("""
    ## How to Use This Dashboard

    1. **Navigation**: Use the tabs at the top to switch between different sections of the dashboard.
    2. **Asteroid Selection**: In each section, you can select an asteroid to view its specific data.
    3. **Trajectory Visuals**: View 3D plots of asteroid trajectories for different models.
    4. **Trajectory Comparisons**: Compare positions, velocities, and differences between models.
    5. **Model Comparisons**: Analyze the performance of different prediction models.

    ## Glossary

    - **AU**: Astronomical Unit, the average distance between the Earth and the Sun
    - **PINN**: Physics-Informed Neural Network
    - **N-body**: N-body simulation
    - **RMSE**: Root Mean Square Error, a measure of model accuracy
    - **Position**: Location in space measured in AU
    - **Velocity**: Speed and direction measured in AU/day
    """)

# Final cleanup
if st.session_state.get('data_loaded'):
    clean_memory()



## markdown text for over view tab
        # This project visualizes the results of predicting asteroid trajectories using three different models: 
        # PINN, N-body, and Transformer. Each model offers a unique approach to trajectory prediction.

        # ## Model Descriptions
        
        # ### PINN (Physics-Informed Neural Network)
        # PINNs combine traditional neural networks with physical laws and constraints. For asteroid trajectory prediction:
        # - They learn from both observational data and known physical equations governing orbital mechanics.
        # - PINNs can interpolate between known data points while respecting physical constraints, potentially offering more accurate predictions in regions with sparse data.

        # The PINN model solves the following equation:
 
        # $$\\frac{d^2\\bold{r}}{dt^2} = -\\frac{GM\\bold{r}}{|\\bold{r}|^3} + \\bold{F}_{other}$$

        # Where:
        # - $\\bold{r}$ is the position vector of the asteroid
        # - $G$ is the gravitational constant
        # - $M$ is the mass of the Sun
        # - $\\bold{F}_{other}$ represents other forces (e.g., from other planets)

        # ### N-body Simulation
        # N-body simulations model the gravitational interactions between multiple celestial bodies:
        # - They calculate the forces between all objects in the system at each time step.
        # - This method is computationally intensive but can provide highly accurate results, especially for complex systems with many interacting bodies.

        # The N-body simulation solves the following system of equations for each body $i$:

        # $$\\frac{d^2\\bold{r}_i}{dt^2} = \\sum_{j \\neq i} G m_j \\frac{\\bold{r}_j - \\bold{r}_i}{|\\bold{r}_j - \\bold{r}_i|^3}$$

        # Where:
        # - $\\bold{r}_i$ is the position vector of body $i$
        # - $m_j$ is the mass of body $j$

        # ### Transformer Model
        # Originally designed for natural language processing, Transformers have been adapted for time series prediction:
        # - They use self-attention mechanisms to identify important features and relationships in sequential data.
        # - For asteroid trajectories, they can potentially capture complex patterns and long-term dependencies in the orbital data.