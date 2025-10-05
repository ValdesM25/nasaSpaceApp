import pandas as pd

# Variable global para guardar el DataFrame
df_almacenado = None
# Variable global para guardar el nombre del modelo
modelo_seleccionado = None


def set_dataframe(df):
    """Guarda el DataFrame."""
    global df_almacenado
    df_almacenado = df

def get_dataframe():
    """Recupera el DataFrame."""
    return df_almacenado

def set_modelo(nuevo_modelo):
    """Guarda el nombre del modelo."""
    global modelo_seleccionado
    modelo_seleccionado = nuevo_modelo

def get_modelo():
    """Recupera el nombre del modelo."""
    return modelo_seleccionado