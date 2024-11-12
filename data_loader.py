import streamlit as st
import pandas as pd
import json
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
from utils import calculate_position_difference

logger = logging.getLogger(__name__)

@st.cache_data(ttl=3600)
def load_asteroid_info():
    """Load asteroid information from JSON file"""
    try:
        with open('asteroid_info.json', 'r') as f:
            data = json.load(f)
            logger.info(f"Successfully loaded asteroid info for {len(data)} asteroids")
            return data
    except FileNotFoundError:
        logger.warning("asteroid_info.json not found, using dummy data")
        return create_dummy_asteroid_info()
    except Exception as e:
        logger.error(f"Error loading asteroid info: {str(e)}")
        return create_dummy_asteroid_info()

@st.cache_data(ttl=3600)
def load_planet_data(asteroid_id, planet):
    """Load planetary data for visualization"""
    try:
        path = Path(f'actual_data/asteroid_{asteroid_id}/{planet.lower()}_data.csv')
        if path.exists():
            data = pd.read_csv(path)
            logger.info(f"Successfully loaded {planet} data for asteroid {asteroid_id}")
            return data
        else:
            logger.warning(f"Planet data file not found: {path}")
            return create_dummy_planet_data(planet)
    except Exception as e:
        logger.error(f"Error loading planet data: {str(e)}")
        return create_dummy_planet_data(planet)

@st.cache_data(ttl=3600)
def load_trajectory_data(asteroid_id, data_type):
    """Load trajectory data for a specific asteroid and model"""
    try:
        if data_type == 'actual':
            path = Path(f'actual_data/asteroid_{asteroid_id}/asteroid_{asteroid_id}_data.csv')
        else:
            model_dirs = {
                'N-body': 'n_body',
                'PINN': 'pinn',
                'Transformer': 'transformer',
                'LSTM': 'lstm'
            }
            path = Path(f'model_data/{data_type}/asteroid_{asteroid_id}/{model_dirs[data_type]}_predicted_trajectory.csv')
        
        if path.exists():
            data = pd.read_csv(path)
            logger.info(f"Successfully loaded {data_type} data for asteroid {asteroid_id}")
            return data
        else:
            logger.warning(f"Data file not found: {path}")
            return create_dummy_trajectory_data()
            
    except Exception as e:
        logger.error(f"Error loading {data_type} data: {str(e)}")
        return create_dummy_trajectory_data()

@st.cache_data(ttl=3600)
def load_trajectory_differences(asteroid_id, model):
    """Load or calculate trajectory differences"""
    try:
        # Try to load pre-computed differences first
        diff_path = Path(f'model_data/{model}/asteroid_{asteroid_id}/{model.lower()}_trajectory_differences.csv')
        
        if diff_path.exists():
            diff_data = pd.read_csv(diff_path)
            logger.info(f"Loaded pre-computed differences for {model}")
            return diff_data
        
        # If pre-computed differences don't exist, calculate them
        actual_data = load_trajectory_data(asteroid_id, 'actual')
        model_data = load_trajectory_data(asteroid_id, model)
        
        if actual_data is not None and model_data is not None:
            diff_data = calculate_position_difference(actual_data, model_data)
            logger.info(f"Calculated differences for {model}")
            return diff_data
        
        return None
        
    except Exception as e:
        logger.error(f"Error loading trajectory differences: {str(e)}")
        return None

def create_dummy_asteroid_info():
    """Create dummy asteroid information"""
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

def create_dummy_trajectory_data():
    """Create dummy trajectory data"""
    t = np.linspace(0, 365*5, 1000)  # 5 years of data
    # Convert to Julian dates starting from J2000
    julian_dates = t + 2451545.0
    
    return pd.DataFrame({
        'time': julian_dates,
        'x': np.sin(2*np.pi*t/365) * 2,
        'y': np.cos(2*np.pi*t/365) * 2,
        'z': np.sin(2*np.pi*t/365) * 0.1,
        'vx': np.cos(2*np.pi*t/365) * 2 * 2*np.pi/365,
        'vy': -np.sin(2*np.pi*t/365) * 2 * 2*np.pi/365,
        'vz': np.cos(2*np.pi*t/365) * 0.1 * 2*np.pi/365
    })

def create_dummy_planet_data(planet):
    """Create dummy planetary orbit data"""
    t = np.linspace(0, 365*5, 1000)  # 5 years of data
    julian_dates = t + 2451545.0
    
    # Different orbital parameters for each planet
    orbital_params = {
        'Mercury': {'a': 0.387, 'period': 88},
        'Venus': {'a': 0.723, 'period': 225},
        'Earth': {'a': 1.0, 'period': 365},
        'Mars': {'a': 1.524, 'period': 687},
        'Jupiter': {'a': 5.203, 'period': 4333}
    }
    
    params = orbital_params.get(planet, {'a': 1.0, 'period': 365})
    a = params['a']  # semi-major axis
    period = params['period']
    
    return pd.DataFrame({
        'time': julian_dates,
        'x': np.sin(2*np.pi*t/period) * a,
        'y': np.cos(2*np.pi*t/period) * a,
        'z': np.sin(2*np.pi*t/period) * a * 0.05  # Small z-variation
    })

def lazy_load_data(asteroid_id):
    """Load all necessary data for an asteroid"""
    if not st.session_state.data_loaded or st.session_state.get('last_asteroid') != asteroid_id:
        try:
            # Load actual trajectory
            actual_data = load_trajectory_data(asteroid_id, 'actual')
            st.session_state.data['actual'] = actual_data
            
            # Load planet data
            planets_data = {}
            for planet in ['Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter']:
                planets_data[planet] = load_planet_data(asteroid_id, planet)
            st.session_state.data['planets'] = planets_data
            
            # Load model predictions and differences
            differences = {}
            for model in ['PINN', 'N-body', 'Transformer', 'LSTM']:
                # Load model predictions
                model_data = load_trajectory_data(asteroid_id, model)
                st.session_state.data[model.lower()] = model_data
                
                # Load or calculate differences
                diff_data = load_trajectory_differences(asteroid_id, model)
                if diff_data is not None:
                    differences[model.lower()] = diff_data
            
            st.session_state.data['differences'] = differences
            st.session_state.data_loaded = True
            st.session_state.last_asteroid = asteroid_id
            logger.info(f"Successfully loaded all data for asteroid {asteroid_id}")
            
        except Exception as e:
            logger.error(f"Error in lazy_load_data: {str(e)}")
            return {}
    
    return st.session_state.data