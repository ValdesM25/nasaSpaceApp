from flask import Flask, render_template

app = Flask(__name__)

# login
@app.route("/")
def home():
    return render_template("login.html", prueba="")


# main
app.run(debug=True)

