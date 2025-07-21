from flask import Flask, render_template, request
import pickle
import os

app = Flask(__name__)

# Load model and vectorizer
model = pickle.load(open("model/cyberbully_model.pkl", "rb"))
vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/services')
def services():
    return render_template("services.html")

@app.route('/contact')
def contact():
    return render_template("contact.html")

@app.route("/detect", methods=["GET", "POST"])
def detect():
    prediction = None
    text = ""

    if request.method == "POST":
        text = request.form["text"]
        vec = vectorizer.transform([text])
        result = model.predict(vec)[0]
        prediction = "Cyberbullying Detected ❌" if result == 1 else "No Cyberbullying ✅"

    return render_template("detect.html", prediction=prediction, text=text)


if __name__ == '__main__':
    app.run(debug=True)