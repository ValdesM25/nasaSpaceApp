import pandas as pd
import numpy as np

def create_efficient_features(df):
    df_eff = df.copy()
    df_eff['period_duration_ratio'] = df_eff['transit_period'] / (df_eff['transit_duration'] + 0.001)
    df_eff['snr_approximation'] = df_eff['transit_depth'] / (df_eff['transit_duration'] + 1)
    expected_depth = (df_eff['planet_radius'] / df_eff['star_radius']) ** 2 * 1e6
    df_eff['depth_discrepancy'] = (df_eff['transit_depth'] - expected_depth) / (expected_depth + 1)
    df_eff['star_luminosity'] = (df_eff['star_radius'] ** 2) * (df_eff['star_temperature'] / 5778) ** 4
    df_eff['transit_probability'] = df_eff['star_radius'] / (df_eff['transit_period'] ** (2/3) + 0.001)
    df_eff['log_transit_depth'] = np.log1p(df_eff['transit_depth'])
    df_eff['log_transit_period'] = np.log1p(df_eff['transit_period'])
    return df_eff