from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import io

from server.memory import retornodf
from ml.inference import predict_exoplanet

app = Flask(__name__)

# login
@app.route("/")
def home():
    return render_template("login.html")
# crear proyecto/cargar desde una plantilla
@app.route("/inicio")
def inicio():
    return render_template("modelo.html", prueba="")
# mandar al modelo
# --- RUTA MODIFICADA ---
# Ahora la ruta acepta una variable de texto llamada 'nombre_modelo'
@app.route("/dashboard/<nombre_modelo>")
def modelo(nombre_modelo):
    # Pasamos la variable 'nombre_modelo' al template de dashboard.html
    # Dentro del HTML, podrás acceder a ella usando {{ modelo_seleccionado }}
    return render_template("dashboard.html", modelo_seleccionado=nombre_modelo)



# --- RUTA /predict MODIFICADA ---
@app.route('/predict', methods=['POST'])
def predict():
    if 'csvfile' not in request.files or request.files['csvfile'].filename == '':
        return "No se seleccionó ningún archivo", 400
    
    file = request.files['csvfile']
    
    # NUEVO: Obtenemos el nombre del modelo desde un campo oculto del formulario
    nombre_modelo_actual = request.form.get('modelo_seleccionado')

    try:
        csv_data = io.StringIO(file.stream.read().decode("UTF8"))
        df = pd.read_csv(csv_data)
        
        headers = df.columns.values.tolist()
        data = df.values.tolist()


        # ####################################################
        # ####################################################
        # guardar en memoria, kepler.csv funciona bien
        # ####################################################
        df_almacenado = df
        print(df_almacenado)

        res = predict_exoplanet(df_almacenado)
        retornodf = res
        # ####################################################
        # llamar al modelo
        # ####################################################

        # NUEVO: Pasamos el nombre del modelo a la plantilla de predicción
        return render_template('predict.html', headers=headers, data=data, modelo_actual=nombre_modelo_actual)
            
    except Exception as e:
        return f"Hubo un error al procesar el archivo: {e}", 500


# --- RUTA /predict MODIFICADA ---
# --- RUTA /predict MODIFICADA ---
# --- RUTA /predict MODIFICADA ---
# --- RUTA /predict MODIFICADA ---
# --- RUTA /predict MODIFICADA ---
# --- RUTA /predict MODIFICADA ---
@app.route('/dfres', methods=['GET'])
def dfres():
    temp = retornodf
    return jsonify(temp)

@app.route("/community")
def community():
    return render_template("community.html")

# main
app.run(debug=True)



#<button type="submit" class="w-full h-14 bg-indigo-600 hover:bg-indigo-700 text-white font-semibold rounded-lg transition #duration-300 shadow-lg hover:shadow-indigo-500/50">Enter Orbit</button>
