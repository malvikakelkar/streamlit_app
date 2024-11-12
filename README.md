# Asteroid Trajectory Dashboard

A comprehensive dashboard for visualizing and comparing different machine learning models' predictions of asteroid trajectories.

## Project Structure

```
asteroid_dashboard/
├── app/
│   ├── __init__.py
│   ├── main.py                  # Main application file
│   ├── utils.py                 # Utility functions
│   ├── data_loading.py          # Data loading functions
│   ├── visualization.py         # Visualization functions
│   └── display_functions.py     # Display components
├── data/
│   ├── actual_data/            # Actual trajectory data
│   │   └── asteroid_{id}/
│   │       ├── asteroid_{id}_data.csv
│   │       └── {planet}_data.csv
│   └── model_data/            # Model predictions
│       ├── PINN/
│       ├── N-body/
│       ├── Transformer/
│       └── LSTM/
├── requirements.txt
└── README.md
```

## Models Overview

The dashboard compares four different approaches for asteroid trajectory prediction:

### 1. Physics-Informed Neural Network (PINN)
- Combines deep learning with physical constraints
- Incorporates Newton's laws directly into the loss function
- Balances data-driven learning with physical principles
- Equation: $$\frac{d^2\mathbf{r}}{dt^2} = -\frac{GM\mathbf{r}}{|\mathbf{r}|^3} + \mathbf{F}_{other}$$

### 2. N-body Gravitational Model
- Traditional physics-based simulation
- Accounts for gravitational interactions between all bodies
- High accuracy but computationally intensive
- Equation: $$\frac{d^2\mathbf{r}_i}{dt^2} = \sum_{j \neq i} G m_j \frac{\mathbf{r}_j - \mathbf{r}_i}{|\mathbf{r}_j - \mathbf{r}_i|^3}$$

### 3. Transformer Model
- Attention-based deep learning architecture
- Captures long-range dependencies in trajectory data
- Effective at learning periodic patterns
- Key components:
  - Multi-head attention
  - Positional encoding
  - Feed-forward networks

### 4. LSTM Model
- Specialized recurrent neural network
- Designed for sequential data prediction
- Maintains internal memory state
- Components:
  - Forget gate
  - Input gate
  - Output gate
  - Memory cell

## Installation

```bash
pip install -r requirements.txt
```

## Running the Dashboard

```bash
streamlit run app/main.py
```

## Data Format

### Trajectory Data
- Time (Julian Date)
- Position (x, y, z) in AU
- Velocity (vx, vy, vz) in AU/day

### Model Predictions
- Predicted positions and velocities
- Difference from actual trajectory
- Performance metrics (RMSE, max error, etc.)

## Features

1. Interactive 3D Visualization
   - Asteroid trajectories
   - Planetary orbits
   - Multiple view angles

2. Model Comparison
   - Side-by-side trajectory visualization
   - Error analysis
   - Performance metrics

3. Detailed Analysis
   - Coordinate-wise comparisons
   - Velocity predictions
   - Error distributions

4. User Interface
   - Model selection
   - Coordinate system toggle
   - Planet orbit visibility