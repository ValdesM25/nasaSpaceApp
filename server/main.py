from flask import Flask, render_template

# import memory

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


@app.route("/community")
def community():
    return render_template("community.html")

# main
app.run(debug=True)


#<button type="submit" class="w-full h-14 bg-indigo-600 hover:bg-indigo-700 text-white font-semibold rounded-lg transition #duration-300 shadow-lg hover:shadow-indigo-500/50">Enter Orbit</button>
