import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os
from perlin_noise import PerlinNoise 
from flask import Flask, render_template, request, jsonify

# --- Inicialización de Flask ---
app = Flask(__name__)

# ----------------------------------------------------------------------
# PASO 1: FUNCIÓN DE CÁLCULO DE PROPIEDADES (Lógica de Python)
# ----------------------------------------------------------------------

def get_planet_properties(transit_depth_ppm, stellar_radius_sol, stellar_effective_temp_k):
    """
    Calcula propiedades estimadas (Radio y Temp) a partir de los 3 parámetros clave.
    """
    transit_depth_fraction = transit_depth_ppm / 1_000_000
    planetary_radius_earth = 1.0 * np.sqrt(transit_depth_fraction) * 109 
    
    planetary_radius_earth = max(1.0, min(10.0, planetary_radius_earth)) 
    
    temp_base = 800 + (planetary_radius_earth - 5.5) * 50
    equilibrium_temperature_k = max(200, min(1500, temp_base))
    
    return {
        "radius_earth": round(planetary_radius_earth, 2),
        "temp_k": round(equilibrium_temperature_k, 2)
    }

# ----------------------------------------------------------------------
# PASO 2: FUNCIÓN DE GENERACIÓN DE MAPA DE RUIDO (Lógica de Python)
# ----------------------------------------------------------------------

def generate_noise_map(width, height, scale, octaves, persistence, lacunarity, seed):
    """Genera un mapa de ruido de Perlin 2D para simular la superficie."""
    
    noise = PerlinNoise(octaves=octaves, seed=seed) 
    world = np.zeros((height, width))
    
    for i in range(height):
        for j in range(width):
            x_norm = j / width
            y_norm = i / height
            world[i][j] = noise([x_norm * scale, y_norm * scale]) 
    
    min_val = np.min(world)
    max_val = np.max(world)
    world = (world - min_val) / (max_val - min_val)
    return world

# ----------------------------------------------------------------------
# PASO 3: APLICACIÓN DE COLOR Y GUARDADO DE LA IMAGEN (Lógica de Python)
# ----------------------------------------------------------------------

def crear_y_guardar_mapa_exoplaneta(parametros, planet_name):
    """Genera el mapa de textura y lo guarda en la carpeta 'Exoplanetas'."""
    
    WIDTH, HEIGHT = 1024, 512 
    NOISE_SCALE = 10.0
    OCTAVES = 6
    PERSISTENCE = 0.5
    LACUNARITY = 2.0
    SEED = sum(ord(c) for c in planet_name) % 1000 

    mapa_ruido = generate_noise_map(WIDTH, HEIGHT, NOISE_SCALE, OCTAVES, PERSISTENCE, LACUNARITY, SEED)

    temp_k = parametros["temp_k"]
    
    if temp_k > 800:
        colors = ["#4D0000", "#FF4500", "#FFD700"] 
        labels = "Mundo de Lava"
    elif temp_k > 373:
        colors = ["#000033", "#CC9966", "#996633"] 
        labels = "Desierto Caliente"
    elif temp_k >= 273 and temp_k <= 373:
        colors = ["#000080", "#228B22", "#D2B48C", "#FFFFFF"] 
        labels = "Templado (Rocas/Agua)"
    else:
        colors = ["#000066", "#ADD8E6", "#FFFFFF"] 
        labels = "Mundo Helado"

    cmap = LinearSegmentedColormap.from_list(labels, colors)

    plt.figure(figsize=(12, 6))
    plt.imshow(mapa_ruido, cmap=cmap)
    plt.xticks([]) 
    plt.yticks([]) 

    carpeta_destino = "Exoplanetas" 
    clean_planet_name = planet_name.replace(' ', '_').replace('.', '')
    nombre_archivo = f"mapa_{clean_planet_name}_R{parametros['radius_earth']}.png"
    ruta_completa = os.path.join(carpeta_destino, nombre_archivo)

    if not os.path.exists(carpeta_destino):
        os.makedirs(carpeta_destino)
        
    plt.savefig(ruta_completa, bbox_inches='tight', pad_inches=0)
    plt.close()

    return ruta_completa

# ----------------------------------------------------------------------
# RUTAS DEL SERVIDOR FLASK
# ----------------------------------------------------------------------

@app.route('/')
def index():
    """Ruta principal: Muestra la interfaz HTML."""
    return render_template('interfaz.html')

@app.route('/calcular', methods=['POST'])
def calcular_propiedades():
    """Ruta AJAX: Calcula R y T cuando el usuario presiona 'Enviar Especificaciones'."""
    data = request.json
    
    try:
        transit_depth_ppm = float(data.get('transit_depth_ppm', 0))
        stellar_radius_sol = float(data.get('stellar_radius_sol', 0))
        stellar_effective_temp_k = float(data.get('stellar_effective_temp_k', 0))
        
        propiedades = get_planet_properties(transit_depth_ppm, stellar_radius_sol, stellar_effective_temp_k)
        
        return jsonify({
            'success': True,
            'radius_earth': propiedades['radius_earth'],
            'temp_k': propiedades['temp_k']
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 400

@app.route('/generar', methods=['POST'])
def generar_imagen():
    """Ruta principal: Genera y guarda la imagen cuando el usuario presiona 'Generar Imagen'."""
    data = request.form
    
    try:
        # Extraer parámetros esenciales para el cálculo
        planet_name = data.get('planetName', 'Exoplaneta_Generado')
        transit_depth_ppm = float(data.get('transit_depth_ppm'))
        stellar_radius_sol = float(data.get('stellar_radius_sol'))
        stellar_effective_temp_k = float(data.get('stellar_effective_temp_k'))
        
        # 1. Calcular propiedades
        propiedades = get_planet_properties(
            transit_depth_ppm,
            stellar_radius_sol,
            stellar_effective_temp_k
        )
        
        # 2. Generar y guardar la imagen
        ruta_guardada = crear_y_guardar_mapa_exoplaneta(propiedades, planet_name)
        
        mensaje = f"Éxito: Imagen de '{planet_name}' generada y guardada en {ruta_guardada}. R: {propiedades['radius_earth']} $R_\oplus$, T: {propiedades['temp_k']} K."
        
        # Regresar a la interfaz con un mensaje de éxito
        return render_template('interfaz.html', success_message=mensaje)

    except Exception as e:
        mensaje = f"Error al generar la imagen: {e}. Asegúrate de que todos los campos numéricos estén correctos."
        return render_template('interfaz.html', error_message=mensaje)

# ----------------------------------------------------------------------
# EJECUCIÓN DEL SERVIDOR
# ----------------------------------------------------------------------
if __name__ == '__main__':
    # Ejecuta el servidor en http://127.0.0.1:5000/
    app.run(debug=True)