from flask import Flask, render_template

app = Flask(__name__)

# login
@app.route("/")
def home():
    return render_template("login.html")
# mandar al modelo
@app.route("/modelo")
def modelo():
    return render_template("modelo.html", prueba="")


# main
app.run(debug=True)


#<button type="submit" class="w-full h-14 bg-indigo-600 hover:bg-indigo-700 text-white font-semibold rounded-lg transition #duration-300 shadow-lg hover:shadow-indigo-500/50">Enter Orbit</button>
