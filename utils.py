import gc
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

def clean_memory():
    """Memory management utility"""
    gc.collect()

def format_mass(mass_str):
    """Format mass values with units"""
    try:
        mass = float(mass_str)
        return f"{mass:.2e} kg"
    except (ValueError, TypeError):
        return mass_str

def julian_to_gregorian(julian_date):
    """Convert Julian date to Gregorian date"""
    try:
        return datetime.fromordinal(int(julian_date - 1721425.5)) + timedelta(days=julian_date % 1)
    except:
        return None

def calculate_error_metrics(actual, predicted):
    """Calculate comprehensive error metrics"""
    try:
        if isinstance(actual, pd.DataFrame) and isinstance(predicted, pd.DataFrame):
            # Calculate Euclidean distance for positions
            position_diff = np.sqrt(
                (actual['x'] - predicted['x'])**2 +
                (actual['y'] - predicted['y'])**2 +
                (actual['z'] - predicted['z'])**2
            )
            
            # Calculate metrics
            rmse = np.sqrt(np.mean(position_diff**2))
            mae = np.mean(np.abs(position_diff))
            max_error = np.max(position_diff)
            
            return {
                'rmse': rmse,
                'mae': mae,
                'max_error': max_error
            }
    except Exception as e:
        print(f"Error calculating metrics: {str(e)}")
        return None

def calculate_position_difference(actual_data, model_data):
    """Calculate position differences between actual and model data"""
    try:
        diff_data = pd.DataFrame()
        diff_data['time'] = actual_data['time']
        
        # Calculate position differences
        for coord in ['x', 'y', 'z']:
            diff_data[f'{coord}_diff'] = actual_data[coord] - model_data[coord]
            
        # Calculate velocity differences if available
        if all(col in actual_data.columns for col in ['vx', 'vy', 'vz']):
            for coord in ['vx', 'vy', 'vz']:
                diff_data[f'{coord[1]}_diff'] = actual_data[coord] - model_data[coord]
        
        return diff_data
    except Exception as e:
        print(f"Error calculating differences: {str(e)}")
        return None

def format_gregorian_date(date):
    """Format Gregorian date for display"""
    try:
        return date.strftime('%Y-%m-%d')
    except:
        return "Invalid date"

def get_date_range(data):
    """Get date range from trajectory data"""
    try:
        start_date = julian_to_gregorian(data['time'].min())
        end_date = julian_to_gregorian(data['time'].max())
        return format_gregorian_date(start_date), format_gregorian_date(end_date)
    except:
        return "N/A", "N/A"