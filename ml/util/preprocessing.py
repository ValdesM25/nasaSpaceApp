import pandas as pd
from .features import create_efficient_features

features = [
    'transit_period', 'transit_duration', 'transit_depth', 'planet_radius',
    'star_temperature', 'star_radius', 'star_surface_gravity', 'star_brightness',
    'period_duration_ratio', 'snr_approximation', 'depth_discrepancy',
    'star_luminosity', 'transit_probability', 'log_transit_depth', 'log_transit_period'
]

def preprocess_input(input_data):
    df = pd.DataFrame([input_data])
    df = create_efficient_features(df)
    df = df[features].fillna(0)
    return df
