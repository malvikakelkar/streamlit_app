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
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Before loading data

logger.info("Starting app initialization...")

@st.cache_data
def load_asteroid_info():
    logger.info("Attempting to load asteroid info...")
    try:
        with open('asteroid_info.json', 'r') as f:
            data = json.load(f)
            logger.info("Successfully loaded asteroid info")
            return data
    except FileNotFoundError:
        logger.error("asteroid_info.json file not found")
        st.error("asteroid_info.json file not found")
        return {}
    except Exception as e:
        logger.error(f"Error loading asteroid info: {str(e)}")
        st.error(f"Error loading asteroid info: {str(e)}")
        return {}

# Set page config
st.set_page_config(
    layout="wide", 
    page_title="Asteroid Trajectory", 
    page_icon="☄️",
    initial_sidebar_state="auto",
    menu_items=None
)

st.markdown("""
<script type="text/x-mathjax-config">
MathJax.Hub.Config({
  tex2jax: {
    inlineMath: [['$','$'], ['\\(','\\)']],
    displayMath: [['$$','$$'], ['\\[','\\]']],
    processEscapes: true,
    processEnvironments: true,
    skipTags: ['script', 'noscript', 'style', 'textarea', 'pre'],
    TeX: { 
      equationNumbers: { autoNumber: "AMS" },
      extensions: ["AMSmath.js", "AMSsymbols.js"],
      Macros: {
        bold: ['{\\bf #1}', 1]
      }
    }
  }
});
MathJax.Hub.Queue(function() {
  var all = MathJax.Hub.getAllJax(), i;
  for(i = 0; i < all.length; i += 1) {
    all[i].SourceElement().parentNode.className += ' has-jax';
  }
});
</script>
<script type="text/javascript" async src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-MML-AM_CHTML"></script>
</script>
<script type="text/javascript" async src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-MML-AM_CHTML"></script>
<style>
    /* Existing styles */
    .MathJax_Display {
        overflow-x: auto;
        overflow-y: hidden;
    }
    .MathJax {
        font-size: 120% !important;
    }
    .has-jax {
        background-color: #f8f9fa;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Add new styles here */
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
    }
    
    h1 {
        font-size: 2.5rem !important;
    }
    
    /* Add this to ensure proper spacing in the layout */
    .stMarkdown div {
        margin-bottom: 0.5rem;
    }
    
    /* Improve metric spacing */
    [data-testid="stMetricValue"] div {
        margin-bottom: 0.25rem !important;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
    <style>
        .block-container {
            padding-top: 1rem;
            padding-bottom: 3rem;  /* Add padding to bottom of page */
            padding-left: 5rem;
            padding-right: 5rem;
        }
        /* Reduce spacing between header elements */
        .stMarkdown {
            margin-bottom: 0;
        }
        /* Reduce spacing in selectbox */
        .stSelectbox {
            margin-bottom: 0.5rem;
        }
        /* Adjust metric spacing */
        [data-testid="stMetricValue"] {
            margin-bottom: 0.5rem;
        }
        /* Reduce tab spacing */
        .stTabs [data-baseweb="tab-list"] {
            gap: 2rem;
            margin-bottom: 0.5rem;
        }
        /* Adjust header margins */
        h1 {
            margin-top: 0 !important;
            margin-bottom: 1rem !important;
        }
        /* Restore original expander (Help section) styling */
        .streamlit-expanderHeader {
            padding: 1rem !important;
            margin-top: 1rem !important;
        }
        .streamlit-expanderContent {
            padding: 1rem !important;
        }
        /* Add bottom margin to the Help section */
        [data-testid="stExpander"] {
            margin-bottom: 2rem;
        }
    
    </style>
""", unsafe_allow_html=True)


# Load asteroid info
@st.cache_data
def load_asteroid_info():
    try:
        with open('asteroid_info.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error("asteroid_info.json file not found. Please make sure it exists in the same directory as this script.")
        return {}

asteroid_info = load_asteroid_info()
asteroid_ids = [16, 21, 243, 253, 433, 4179, 25143, 101955, 162173]

def load_trajectory_data(asteroid_id, data_type):
    if data_type == 'actual':
        path = Path(f'actual_data/asteroid_{asteroid_id}/asteroid_{asteroid_id}_data.csv')
    else:
        if data_type == 'N-body':
            model_dir = 'N-body'
            file_name = f'n_body_predicted_trajectory.csv'
        elif data_type == 'PINN':
            model_dir = 'PINN'
            file_name = f'pinn_predicted_trajectory.csv'
        elif data_type == 'Transformer':
            model_dir = 'Transformer'
            file_name = f'transformer_predicted_trajectory.csv'
        else:
            st.error(f"Unknown data type: {data_type}")
            return None

        path = Path(f'model_data/{model_dir}/asteroid_{asteroid_id}/{file_name}')
    
    try:
        if not path.exists():
            print(f"File not found: {path}")
            return None
        data = pd.read_csv(path)
        print(f"Loaded {data_type} trajectory data for Asteroid {asteroid_id}. Columns: {data.columns}")
        return data
    except Exception as e:
        print(f"Error loading trajectory data for {data_type} model and Asteroid {asteroid_id}: {str(e)}")
        return None
    
def load_analysis_summary(asteroid_id, model):
    if model == "N-body":
        file_name = f"n_body_analysis_summary.json"
    elif model == "PINN":
        file_name = f"pinn_analysis_summary.json"
    elif model == "Transformer":
        file_name = f"transformer_analysis_summary.json"
    else:
        st.error(f"Unknown model: {model}")
        return None

    file_path = Path(f'model_data/{model}/asteroid_{asteroid_id}/{file_name}')
    try:
        if not file_path.exists():
            st.warning(f"File not found: {file_path}")
            return None
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading analysis summary for {model} model and Asteroid {asteroid_id}: {str(e)}")
        return None

@st.cache_data
def load_planet_data(asteroid_id, planet):
    path = Path(f'actual_data/asteroid_{asteroid_id}/{planet}_data.csv')
    return pd.read_csv(path)


@st.cache_data
def load_trajectory_differences(asteroid_id, model):
    if model == 'N-body':
        file_name = f'n_body_trajectory_differences.csv'
    elif model == 'PINN':
        file_name = f'pinn_trajectory_differences.csv'
    elif model == 'Transformer':
        file_name = f'transformer_trajectory_differences.csv'
    else:
        st.error(f"Unknown model: {model}")
        return None

    path = Path(f'model_data/{model}/asteroid_{asteroid_id}/{file_name}')
    try:
        if not path.exists():
            st.warning(f"File not found: {path}")
            return None
        data = pd.read_csv(path)
        # Validate the structure of the loaded data
        expected_columns = ['time', 'x_diff', 'y_diff', 'z_diff', 'vx_diff', 'vy_diff', 'vz_diff']
        if not all(col in data.columns for col in expected_columns):
            st.error(f"Invalid data structure for {model} model and Asteroid {asteroid_id}. Expected columns: {expected_columns}")
            return None
        return data
    except Exception as e:
        st.error(f"Error loading trajectory differences for {model} model and Asteroid {asteroid_id}: {str(e)}")
        return None

def julian_to_gregorian(julian_date):
    return datetime.fromordinal(int(julian_date - 1721425.5)) + timedelta(days=julian_date % 1)


def load_analysis_summary(asteroid_id, model):
    if model == "N-body":
        file_name = f"n_body_analysis_summary.json"
    elif model == "PINN":
        file_name = f"pinn_analysis_summary.json"
    elif model == "Transformer":
        file_name = f"transformer_analysis_summary.json"
    else:
        st.error(f"Unknown model: {model}")
        return None

    file_path = Path(f'model_data/{model}/asteroid_{asteroid_id}/{file_name}')
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error(f"Analysis summary file not found for {model} model and Asteroid {asteroid_id}")
        return None

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
        return mass_str  # return as is if it's not a number

def display_model_analysis(col, model_name, summary):
    with col:
        # Use Streamlit's built-in card container which respects theme
        with st.container():
            st.subheader(f"{model_name} Model Analysis")
            st.write(f"RMSE Position: {format_number(summary.get('rmse_position', 'N/A'))}")
            st.write(f"RMSE Velocity: {format_number(summary.get('rmse_velocity', 'N/A'))}")
            
            st.write("Deviations")
            
            # Create a DataFrame for deviations to use Streamlit's native table display
            deviations_data = {
                '': ['Min Position', 'Min Velocity', 'Max Position', 'Max Velocity'],
                'X': [
                    format_number(summary['minimum_deviations']['position'].get('X_diff', 'N/A')),
                    format_number(summary['minimum_deviations']['velocity'].get('VX_diff', 'N/A')),
                    format_number(summary['maximum_deviations']['position'].get('X_diff', 'N/A')),
                    format_number(summary['maximum_deviations']['velocity'].get('VX_diff', 'N/A'))
                ],
                'Y': [
                    format_number(summary['minimum_deviations']['position'].get('Y_diff', 'N/A')),
                    format_number(summary['minimum_deviations']['velocity'].get('VY_diff', 'N/A')),
                    format_number(summary['maximum_deviations']['position'].get('Y_diff', 'N/A')),
                    format_number(summary['maximum_deviations']['velocity'].get('VY_diff', 'N/A'))
                ],
                'Z': [
                    format_number(summary['minimum_deviations']['position'].get('Z_diff', 'N/A')),
                    format_number(summary['minimum_deviations']['velocity'].get('VZ_diff', 'N/A')),
                    format_number(summary['maximum_deviations']['position'].get('Z_diff', 'N/A')),
                    format_number(summary['maximum_deviations']['velocity'].get('VZ_diff', 'N/A'))
                ]
            }
            deviations_df = pd.DataFrame(deviations_data)
            st.dataframe(deviations_df.set_index(''), hide_index=False)

            st.write("Maximum Absolute Differences")
            max_diff_data = {
                '': ['Position', 'Velocity'],
                'X': [
                    format_number(summary['max_absolute_differences'].get('X_diff', 'N/A')),
                    format_number(summary['max_absolute_differences'].get('VX_diff', 'N/A'))
                ],
                'Y': [
                    format_number(summary['max_absolute_differences'].get('Y_diff', 'N/A')),
                    format_number(summary['max_absolute_differences'].get('VY_diff', 'N/A'))
                ],
                'Z': [
                    format_number(summary['max_absolute_differences'].get('Z_diff', 'N/A')),
                    format_number(summary['max_absolute_differences'].get('VZ_diff', 'N/A'))
                ]
            }
            max_diff_df = pd.DataFrame(max_diff_data)
            st.dataframe(max_diff_df.set_index(''), hide_index=False)


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
    
    diff_data['time'] = pd.to_datetime(diff_data['time'], unit='D', origin='julian')
    
    colors = ['darkred', 'darkgreen', 'darkblue', 'darkorange', 'purple', 'teal']
    
    for i, col in enumerate(available_cols):
        row, col_num = i // 3 + 1, i % 3 + 1
        fig.add_trace(go.Scatter(x=diff_data['time'], y=diff_data[col], mode='lines', 
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


def create_model_comparison_pdf(asteroid_ids):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    elements = []

    data = [['Asteroid ID', 'PINN RMSE', 'N-body RMSE', 'Transformer RMSE']]
    
    for asteroid_id in asteroid_ids:
        pinn_summary = load_analysis_summary(asteroid_id, "PINN")
        nbody_summary = load_analysis_summary(asteroid_id, "N-body")
        transformer_summary = load_analysis_summary(asteroid_id, "Transformer")
        
        if all([pinn_summary, nbody_summary, transformer_summary]):
            data.append([
                str(asteroid_id),
                f"Pos: {format_number(pinn_summary['rmse_position'])}\nVel: {format_number(pinn_summary['rmse_velocity'])}",
                f"Pos: {format_number(nbody_summary['rmse_position'])}\nVel: {format_number(nbody_summary['rmse_velocity'])}",
                f"Pos: {format_number(transformer_summary['rmse_position'])}\nVel: {format_number(transformer_summary['rmse_velocity'])}"
            ])

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
    doc.build(elements)
    buffer.seek(0)
    return buffer


def create_time_lapse_animation(asteroid_data, planet_data):
    # Create figure
    fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'scatter3d'}]])

    # Add Sun
    fig.add_trace(go.Scatter3d(x=[0], y=[0], z=[0], mode='markers', 
                               marker=dict(size=10, color='yellow'), name='Sun'))

    # Add asteroid
    fig.add_trace(go.Scatter3d(x=asteroid_data['x'], y=asteroid_data['y'], z=asteroid_data['z'],
                               mode='lines+markers', name='Asteroid', 
                               line=dict(color='cyan', width=4),
                               marker=dict(size=5, color='cyan')))

    # Add planets
    colors = {'Mercury': 'fuchsia', 'Venus': 'orange', 'Earth': 'seagreen', 'Mars': 'red', 'Jupiter': 'gold'}
    for planet, color in colors.items():
        planet_df = planet_data[planet]
        fig.add_trace(go.Scatter3d(x=planet_df['x'], y=planet_df['y'], z=planet_df['z'],
                                   mode='lines+markers', name=planet, 
                                   line=dict(color=color, width=2),
                                   marker=dict(size=3, color=color)))

    # Set up frames for animation
    frames = []
    for i in range(len(asteroid_data)):
        frame = go.Frame(
            data=[
                go.Scatter3d(x=[0], y=[0], z=[0]),  # Sun
                go.Scatter3d(x=asteroid_data['x'][:i+1], y=asteroid_data['y'][:i+1], z=asteroid_data['z'][:i+1]),  # Asteroid
            ] + [
                go.Scatter3d(x=planet_data[planet]['x'][:i+1], y=planet_data[planet]['y'][:i+1], z=planet_data[planet]['z'][:i+1])
                for planet in colors.keys()
            ],
            name=str(i)
        )
        frames.append(frame)

    fig.frames = frames

    # Set up slider
    sliders = [dict(
        steps=[
            dict(method='animate', args=[[str(i)], dict(mode='immediate', frame=dict(duration=50, redraw=True), transition=dict(duration=0))], label=str(i))
            for i in range(0, len(frames), 10)
        ],
        transition=dict(duration=0),
        x=0,
        y=0,
        currentvalue=dict(font=dict(size=12), prefix='Time: ', visible=True, xanchor='center'),
        len=1.0
    )]

    # Update layout
    fig.update_layout(
        scene=dict(
            xaxis_title='X (AU)',
            yaxis_title='Y (AU)',
            zaxis_title='Z (AU)',
            aspectmode='cube'
        ),
        updatemenus=[dict(
            type='buttons',
            showactive=False,
            buttons=[
                dict(label='Play',
                     method='animate',
                     args=[None, dict(frame=dict(duration=50, redraw=True), fromcurrent=True, transition=dict(duration=0))]),
                dict(label='Pause',
                     method='animate',
                     args=[[None], dict(frame=dict(duration=0, redraw=False), mode='immediate', transition=dict(duration=0))])
            ]
        )],
        sliders=sliders
    )

    return fig

# Initialize session state for selected asteroid if it doesn't exist
if 'selected_asteroid' not in st.session_state:
    st.session_state.selected_asteroid = asteroid_ids[0]


# Replace the existing header section with this enhanced info card layout
st.write("""
<div style='padding: 0.5rem 0; margin-bottom: 0.5rem'>
    <div style='display: flex; gap: 1rem; align-items: center'>
        <h1 style='margin: 0; padding: 0'>☄️ Asteroid Trajectory Dashboard</h1>
        <div style='flex-grow: 1'>
""", unsafe_allow_html=True)

# Reorganize the info display to be more compact
col1, col2, col3 = st.columns([2, 2, 2])
with col1:
    if 'selected_asteroid' not in st.session_state:
        st.session_state.selected_asteroid = asteroid_ids[0]
    st.session_state.selected_asteroid = st.selectbox(
        "Select Asteroid",
        asteroid_ids,
        index=asteroid_ids.index(st.session_state.selected_asteroid),
        key="global_asteroid_select",
        label_visibility="collapsed"  # Hide label to save space
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

    # Combine metrics into a single row with less spacing
    metrics_cols = st.columns(3)
    with metrics_cols[0]:
        st.metric("Aphelion", f"{info['aphelion']} AU")
    with metrics_cols[1]:
        st.metric("Perihelion", f"{info['perihelion']} AU")
    with metrics_cols[2]:
        st.metric("Eccentricity", info['eccentricity'])

st.write("</div></div></div>", unsafe_allow_html=True)

# Navigation tabs
tab_overview, tab_visuals, tab_comparisons, tab_models = st.tabs(["Overview", "Trajectory Visuals", "Trajectory Comparisons", "Model Comparisons"])


with tab_overview:
    st.header("Overview")
    
    col_intro = st.container()
    
    with col_intro:
        st.markdown("""
        This project visualizes the results of predicting asteroid trajectories using three different models: PINN, N-body, and Transformer. Each model offers a unique approach to trajectory prediction.

        ## Model Descriptions
        
        ### PINN (Physics-Informed Neural Network)
        PINNs combine traditional neural networks with physical laws and constraints. For asteroid trajectory prediction:
        - They learn from both observational data and known physical equations governing orbital mechanics.
        - PINNs can interpolate between known data points while respecting physical constraints, potentially offering more accurate predictions in regions with sparse data.

        The PINN model solves the following equation:
 
        $$\\frac{d^2\\bold{r}}{dt^2} = -\\frac{GM\\bold{r}}{|\\bold{r}|^3} + \\bold{F}_{other}$$

        Where:
        - $\\bold{r}$ is the position vector of the asteroid
        - $G$ is the gravitational constant
        - $M$ is the mass of the Sun
        - $\\bold{F}_{other}$ represents other forces (e.g., from other planets)

        The loss function of the PINN combines data loss and physics loss:

        1. **Data Loss** ($\\mathcal{L}_{\\text{data}}$):
           Measures the discrepancy between the predicted positions and observed data.
    
           $$\\mathcal{L}_{\\text{data}} = \\frac{1}{N_{\\text{data}}} \\sum_{k=1}^{N_{\\text{data}}} \\left\\| \\bold{r}_{\\text{pred}}(t_k) - \\bold{r}_{\\text{obs}}(t_k) \\right\\|^2$$

        2. **Physics Loss** ($\\mathcal{L}_{\\text{physics}}$):
           Ensures the network's predictions satisfy the governing differential equations.
    
           $$\\mathcal{L}_{\\text{physics}} = \\frac{1}{N_{\\text{physics}}} \\sum_{j=1}^{N_{\\text{physics}}} \\left\\| \\frac{d^2 \\bold{r}_{\\text{pred}}(t_j)}{dt^2} + G \\sum_{i=1}^N \\frac{M_i (\\bold{r}_{\\text{pred}}(t_j) - \\bold{r}_i(t_j))}{\\|\\bold{r}_{\\text{pred}}(t_j) - \\bold{r}_i(t_j)\\|^3} \\right\\|^2$$

        3. **Total Loss** ($\\mathcal{L}$):
    
           $$\\mathcal{L} = \\mathcal{L}_{\\text{data}} + \\lambda \\mathcal{L}_{\\text{physics}}$$
    
           Where $\\lambda$ is a hyperparameter balancing data fidelity and physics conformity.

        ### N-body Simulation
        N-body simulations model the gravitational interactions between multiple celestial bodies:
        - They calculate the forces between all objects in the system at each time step.
        - This method is computationally intensive but can provide highly accurate results, especially for complex systems with many interacting bodies.

        The N-body simulation solves the following system of equations for each body $i$:

        $$\\frac{d^2\\bold{r}_i}{dt^2} = \\sum_{j \\neq i} G m_j \\frac{\\bold{r}_j - \\bold{r}_i}{|\\bold{r}_j - \\bold{r}_i|^3}$$

        Where:
        - $\\bold{r}_i$ is the position vector of body $i$
        - $m_j$ is the mass of body $j$

        ### Transformer Model
        Originally designed for natural language processing, Transformers have been adapted for time series prediction:
        - They use self-attention mechanisms to identify important features and relationships in sequential data.
        - For asteroid trajectories, they can potentially capture complex patterns and long-term dependencies in the orbital data.

        The core of the Transformer model is the self-attention mechanism, which can be expressed as:

        $$\\text{Attention}(Q, K, V) = \\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right)V$$

        Where:
        - $Q$, $K$, and $V$ are query, key, and value matrices
        - $d_k$ is the dimension of the key vectors

        Each model has its strengths and weaknesses in predicting asteroid trajectories, and this dashboard allows for detailed comparisons of their performance.
        """)



with tab_visuals:
    st.header("Trajectory Visuals")
    st.markdown("*Double-click any legend item to isolate that trajectory, or hide others*")

    selected_asteroid = st.session_state.selected_asteroid


    actual_data = load_trajectory_data(selected_asteroid, 'actual')
    pinn_data = load_trajectory_data(selected_asteroid, 'PINN')
    nbody_data = load_trajectory_data(selected_asteroid, 'N-body')
    transformer_data = load_trajectory_data(selected_asteroid, 'Transformer')

    planets = ['Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter']
    planet_data = {planet: load_planet_data(selected_asteroid, planet) for planet in planets}

    visual_tabs = st.tabs(["Actual Trajectory", "PINN Model", "N-body Model", "Transformer Model"])

    for i, (tab, data, model_name) in enumerate(zip(visual_tabs, [actual_data, pinn_data, nbody_data, transformer_data], 
                                                    ["Actual", "PINN", "N-body", "Transformer"])):
        with tab:
            # st.subheader(f"{model_name} Trajectory")
            fig = create_3d_plot(data, planet_data, f"{model_name} Trajectory of Asteroid {selected_asteroid}")
            st.plotly_chart(fig, use_container_width=True)

with tab_comparisons:
    st.header("Trajectory Comparisons")
    st.markdown("*Double-click any legend item to isolate that trajectory, or hide others*")
    selected_asteroid = st.session_state.selected_asteroid

    
    actual_data = load_trajectory_data(selected_asteroid, 'actual')
    pinn_data = load_trajectory_data(selected_asteroid, 'PINN')
    nbody_data = load_trajectory_data(selected_asteroid, 'N-body')
    transformer_data = load_trajectory_data(selected_asteroid, 'Transformer')
    
    # Add these lines here
    pinn_diff = load_trajectory_differences(selected_asteroid, 'PINN')
    nbody_diff = load_trajectory_differences(selected_asteroid, 'N-body')
    transformer_diff = load_trajectory_differences(selected_asteroid, 'Transformer')
    
    comparison_tabs = st.tabs(["Position Comparisons", "Velocity Comparisons", "Trajectory Differences"])
    
    # Rest of your comparison code...
    with comparison_tabs[0]:
        st.subheader("Position Comparisons")
        for coord in ['x', 'y', 'z']:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.plotly_chart(create_2d_comparison_plot(actual_data, pinn_data, f"PINN vs Actual - {coord.upper()}", coord), use_container_width=True, key=f"pos_pinn_{coord}")
            with col2:
                st.plotly_chart(create_2d_comparison_plot(actual_data, nbody_data, f"N-body vs Actual - {coord.upper()}", coord), use_container_width=True, key=f"pos_nbody_{coord}")
            with col3:
                st.plotly_chart(create_2d_comparison_plot(actual_data, transformer_data, f"Transformer vs Actual - {coord.upper()}", coord), use_container_width=True, key=f"pos_transformer_{coord}")
        
    with comparison_tabs[1]:
        st.subheader("Velocity Comparisons")
        for coord in ['vx', 'vy', 'vz']:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.plotly_chart(create_2d_comparison_plot(actual_data, pinn_data, f"PINN vs Actual - {coord.upper()}", coord), use_container_width=True, key=f"vel_pinn_{coord}")
            with col2:
                st.plotly_chart(create_2d_comparison_plot(actual_data, nbody_data, f"N-body vs Actual - {coord.upper()}", coord), use_container_width=True, key=f"vel_nbody_{coord}")
            with col3:
                st.plotly_chart(create_2d_comparison_plot(actual_data, transformer_data, f"Transformer vs Actual - {coord.upper()}", coord), use_container_width=True, key=f"vel_transformer_{coord}")


with comparison_tabs[2]:
    st.subheader("Trajectory Differences")

    
    for model, diff_data in [("PINN", pinn_diff), ("N-body", nbody_diff), ("Transformer", transformer_diff)]:
        if diff_data is not None and not diff_data.empty:
            # st.write(f"Columns in {model} difference data: {', '.join(diff_data.columns)}")
            fig = create_trajectory_difference_plot(diff_data, f"{model} Trajectory Differences - Asteroid {selected_asteroid}")
            if fig is not None:
                st.plotly_chart(fig, key=f"diff_{model}")
        else:
            st.error(f"No difference data available for {model} model and Asteroid {selected_asteroid}")

with tab_models:
    st.header("Model Comparisons")
    selected_asteroid = st.session_state.selected_asteroid


    pdf_buffer = create_model_comparison_pdf(asteroid_ids)
    st.download_button(
        label="Download Model Comparison PDF",
        data=pdf_buffer,
        file_name="model_comparison.pdf",
        mime="application/pdf"
    )

    # Load analysis summaries for each model
    pinn_summary = load_analysis_summary(selected_asteroid, "PINN")
    nbody_summary = load_analysis_summary(selected_asteroid, "N-body")
    transformer_summary = load_analysis_summary(selected_asteroid, "Transformer")

    # Create columns for each model
    col1, col2, col3 = st.columns(3)

    if pinn_summary:
        display_model_analysis(col1, "PINN", pinn_summary)
    if nbody_summary:
        display_model_analysis(col2, "N-body", nbody_summary)
    if transformer_summary:
        display_model_analysis(col3, "Transformer", transformer_summary)

    # In the Models tab, replace the existing RMSE comparison section with:
# Update the RMSE comparison section
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
            
            # Update layout with fixed grid spacing
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
                    dtick=0.30103,  # This creates evenly spaced grid lines on log scale (log10(2))
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='LightGray'
                ),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            
            # Remove vertical grid lines
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
            
            # Update layout with fixed grid spacing
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
                    dtick=0.30103,  # This creates evenly spaced grid lines on log scale
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='LightGray'
                ),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            
            # Remove vertical grid lines
            fig_velocity.update_xaxes(showgrid=False)
            
            st.plotly_chart(fig_velocity, use_container_width=True)

        # Add explanation text based on selected metric
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



    # Interactive time series plot
    st.subheader("Model Comparisons Over Time")
    st.markdown("*Double-click any legend item to isolate that model, or hide others*")

    coord = st.selectbox("Select coordinate", ['x', 'y', 'z'])
    
    fig = go.Figure()
    
    if all([actual_data is not None, pinn_data is not None, nbody_data is not None, transformer_data is not None]):
        fig.add_trace(go.Scatter(x=actual_data['time'], y=actual_data[coord], mode='lines', name='Actual'))
        fig.add_trace(go.Scatter(x=pinn_data['time'], y=pinn_data[coord], mode='lines', name='PINN'))
        fig.add_trace(go.Scatter(x=nbody_data['time'], y=nbody_data[coord], mode='lines', name='N-body'))
        fig.add_trace(go.Scatter(x=transformer_data['time'], y=transformer_data[coord], mode='lines', name='Transformer'))

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

# Initialize variables to store data for each model
actual_data = None
pinn_data = None
nbody_data = None
transformer_data = None

# Error handling and data validation
def validate_data(data, expected_columns):
    missing_columns = set(expected_columns) - set(data.columns)
    if missing_columns:
        st.error(f"Missing columns in data: {', '.join(missing_columns)}")
        return False
    return True

try:
    # Validate data for each model
    expected_columns = ['time', 'x', 'y', 'z', 'vx', 'vy', 'vz']
    for model in ['actual', 'PINN', 'N-body', 'Transformer']:
        data = load_trajectory_data(selected_asteroid, model)
        if not validate_data(data, expected_columns):
            st.error(f"Data validation failed for {model} model.")
            st.stop()
        
        # Store data in the appropriate variable
        if model == 'actual':
            actual_data = data
        elif model == 'PINN':
            pinn_data = data
        elif model == 'N-body':
            nbody_data = data
        elif model == 'Transformer':
            transformer_data = data

    # Additional data checks
    if actual_data is None or pinn_data is None or nbody_data is None or transformer_data is None:
        st.error("One or more datasets are empty. Please check your data files.")
        st.stop()

except Exception as e:
    st.error(f"An error occurred while loading or processing data: {str(e)}")
    st.stop()

# Add a help section
with st.expander("Help & Information"):
    st.markdown("""
    How to Use This Dashboard

    1. **Navigation**: Use the tabs at the top to switch between different sections of the dashboard.
    2. **Asteroid Selection**: In each section, you can select an asteroid to view its specific data.
    3. **Trajectory Visuals**: View 3D plots of asteroid trajectories for different models.
    4. **Trajectory Comparisons**: Compare positions, velocities, and differences between models.
    5. **Model Comparisons**: Analyze the performance of different prediction models.


    ## Glossary

    - **AU**: Astronomical Unit, the average distance between the Earth and the Sun.
    - **PINN**: Physics-Informed Neural Network
    - **N-body**: N-body simulation
    - **MSE**: Mean Squared Error, a measure of model accuracy

    """)