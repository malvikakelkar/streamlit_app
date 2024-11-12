import streamlit as st

# Updated color scheme with brighter, more visible colors
MODEL_CONFIGS = {
    'PINN': {
        'color': '#FF1493',  # Deep pink
        'file_prefix': 'pinn',
        'description': 'Physics-Informed Neural Network',
        'equations': [
            r"\frac{d^2\mathbf{r}}{dt^2} = -\frac{GM\mathbf{r}}{|\mathbf{r}|^3} + \mathbf{F}_{other}",
            r"\mathcal{L} = \mathcal{L}_{data} + \lambda\mathcal{L}_{physics}"
        ]
    },
    'N-body': {
        'color': '#00FF00',  # Bright green
        'file_prefix': 'n_body',
        'description': 'Gravitational N-body Simulation',
        'equations': [
            r"\frac{d^2\mathbf{r}_i}{dt^2} = \sum_{j \neq i} G m_j \frac{\mathbf{r}_j - \mathbf{r}_i}{|\mathbf{r}_j - \mathbf{r}_i|^3}"
        ]
    },
    'Transformer': {
        'color': '#00BFFF',  # Deep sky blue
        'file_prefix': 'transformer',
        'description': 'Deep Learning Sequence Model',
        'architecture': [
            'Multi-head attention layers',
            'Position encodings',
            'Feed-forward networks'
        ]
    },
    'LSTM': {
        'color': '#FFA500',  # Bright orange
        'file_prefix': 'lstm',
        'description': 'Long Short-Term Memory Network',
        'architecture': [
            'Forget gate',
            'Input gate',
            'Output gate',
            'Memory cell'
        ]
    }
}

# Color scheme for celestial bodies
CELESTIAL_COLORS = {
    'Sun': '#FFD700',  # Gold
    'Mercury': '#9370DB',  # Medium purple
    'Venus': '#FFA07A',    # Light salmon
    'Earth': '#1E90FF',    # Dodger blue
    'Mars': '#FF4500',     # Orange red
    'Jupiter': '#DAA520',  # Golden rod
    'Asteroid': '#00FF00'  # Bright green
}

# Plot styling
PLOT_STYLE = {
    'background_color': '#000000',  # Black background
    'grid_color': '#2F4F4F',        # Dark slate gray grid
    'text_color': '#FFFFFF',        # White text
    'line_width': 3,                # Thicker lines for better visibility
}

def initialize_config():
    """Initialize Streamlit page configuration"""
    st.set_page_config(
        layout="wide",
        page_title="Asteroid Trajectory",
        page_icon="☄️",
    )
    
    # Add custom CSS
    st.markdown("""
        <style>
        .asteroid-header {
            padding: 1rem;
            background-color: #1E1E1E;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
            color: white;
        }
        .stMetricValue {
            font-size: 1.2rem !important;
            color: #00FF00 !important;
        }
        .plot-container {
            background-color: #000000;
            border-radius: 0.5rem;
            padding: 1rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        </style>
    """, unsafe_allow_html=True)