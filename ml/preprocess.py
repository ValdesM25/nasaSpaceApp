import pandas as pd 
import numpy as np
from columnas import *

# Generate error columns
def generate_errors(df, cols, drop=True):
    for col in cols:
        
        patterns = [
            (f'{col}err1', f'{col}err2'),
            (f'{col}_err1', f'{col}_err2'),
        ]
        
        e1, e2 = None, None
        for p1, p2 in patterns:
            if p1 in df.columns and p2 in df.columns:
                e1, e2 = df[p1], df[p2]
                if drop:
                    df.drop(columns=[p1, p2], inplace=True)
                break
        
        if e1 is None or e2 is None:
            continue

        df[f'{col}_avgerr'] = (e1.abs() + e2.abs()) / 2
        df[f'{col}_relerr'] = df[f'{col}_avgerr'] / df[col].replace(0, np.nan)

    return df

# Preprocess function
def preprocess(df, used_columns, cols_dict, row_query=None):
    df = df.rename(columns=cols_dict)

    df = df[used_columns]
    df = generate_errors(df, used_columns)
    if row_query: df = df.query(row_query)

    df['star_brightness'] = (df['star_brightness'] - df['star_brightness'].mean()) / df['star_brightness'].std()

    return df

used_columns = ['category', 'transit_period', 'transit_duration', 'transit_depth', 'transit_depth', 'planet_radius', 'star_temperature', 'star_radius', 'star_surface_gravity', 'star_brightness']

# Kepler
kepler_raw = pd.read_csv('ml/datasets/kepler_objects_of_interest_koi.csv')
kepler_used_columns = ['koi_disposition', 'koi_period', 'koi_duration', 'koi_depth', 'koi_prad', 'koi_steff', 'koi_srad', 'koi_slogg', 'koi_kepmag']
kepler = preprocess(kepler_raw, used_columns, kepler_cols_dict)

# Tess
tess_raw = pd.read_csv('ml/datasets/tess_objects_of_interest_toi.csv')
tess_used_columns = ['tfopwg_disp', 'pl_orbper', 'pl_trandurh', 'pl_trandep', 'pl_rade', 'st_teff', 'st_rad', 'st_logg', 'st_tmag']
tess = preprocess(tess_raw, used_columns, tess_cols_dict)

# K2
k2_raw = pd.read_csv('ml/datasets/k2_planets_and_candidates.csv')
k2_raw['pl_trandep'] *= 10000
k2_raw['pl_trandeperr1'] *= 10000
k2_raw['pl_trandeperr2'] *= 10000

k2_used_columns = ['disposition', 'pl_orbper', 'pl_trandur', 'pl_trandep', 'pl_rade', 'st_teff', 'st_rad', 'st_logg', 'sy_kepmag']
k2 = preprocess(k2_raw, used_columns, k2_cols_dict)

# Save datasets
kepler.to_csv('ml/datasets/kepler.csv', index=False)
tess.to_csv('ml/datasets/tess.csv', index=False)
k2.to_csv('ml/datasets/k2.csv', index=False)