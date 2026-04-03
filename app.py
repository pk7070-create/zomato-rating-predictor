from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load model & scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    # Get data from form
    votes = int(request.form["votes"])
    cost = int(request.form["cost"])
    online_order = int(request.form["online_order"])
    book_table = int(request.form["book_table"])

    rtype = request.form["type"]

    # One hot encoding
    type_dining = 1 if rtype == "Dining" else 0
    type_cafes = 1 if rtype == "Cafes" else 0
    type_buffet = 1 if rtype == "Buffet" else 0
    type_other = 1 if rtype == "other" else 0

    # Model input
    data = np.array([[online_order, book_table, votes, cost,
                      type_buffet, type_cafes, type_dining, type_other]])

    data = scaler.transform(data)
    prediction = model.predict(data)

    return render_template("index.html", result=round(prediction[0], 2))

if __name__ == "__main__":
    app.run(debug=True)

#flask is connected    