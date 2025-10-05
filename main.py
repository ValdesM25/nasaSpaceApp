
from flask import Flask, render_template, request, redirect, url_for, jsonify
import pandas as pd
import io
from flask_cors import CORS # Importa la librería

#from server.memory import retornodf
from ml.inference import predict_exoplanet

# retorno modelo df
retornodf = 7

app = Flask(
    __name__,
    template_folder="server/templates",  # relative path from run.py
    static_folder="server/static"        # relative path from run.py
)

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
        global retornodf
        csv_data = io.StringIO(file.stream.read().decode("UTF8"))
        df = pd.read_csv(csv_data)
        
        headers = df.columns.values.tolist()
        data = df.values.tolist()


        # ####################################################
        # ####################################################
        # guardar en memoria, kepler.csv funciona bien
        # ####################################################
        df_almacenado = df

        res = predict_exoplanet(df_almacenado, 'random_forest')
        print("res", res)
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
@app.route('/dfres')
def dfres():
    #temp = retornodf
    #print('!!!!!!!!!!!!!!!!', temp)
    
    return str(retornodf)
    #return render_template('predict.html', data=retornodf, )

@app.route("/community")
def community():
    return render_template("community.html")
@app.route("/results")
def results():
    json_data = retornodf
    
    if not json_data:
        return "No hay datos para mostrar", 400

    # Extraemos la lista de resultados
    resultados = json_data.get("results", [])  

    if not resultados:
        return "No hay resultados para mostrar", 400

    # Tomamos los headers de la primera fila
    headers = list(resultados[0].keys())

    modelo_actual = json_data.get("model", "Modelo desconocido")

    return render_template("resultscsv.html",
                           headers=headers,
                           data=json_data,
                           modelo=modelo_actual)



# main
app.run(debug=True)



#ruta sistema solar
# Le da permiso al frontend de Vite para hablar con Flask
CORS(app, resources={r"/api/*": {"origins": "http://localhost:5173"}}) 

@app.route('/api/datos_planetas', methods=['GET'])
def get_planetas():
    # Aquí obtienes los datos de tu base de datos o lógica de Flask
    datos = {"planetas": ["Marte", "Tierra", "Júpiter"], "version": "1.0"}
    return jsonify(datos)

#<button type="submit" class="w-full h-14 bg-indigo-600 hover:bg-indigo-700 text-white font-semibold rounded-lg transition #duration-300 shadow-lg hover:shadow-indigo-500/50">Enter Orbit</button>


