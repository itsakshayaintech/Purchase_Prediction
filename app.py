from flask import Flask, render_template, request
import numpy as np
import pickle
import webbrowser

app = Flask(__name__)

# LOAD MODEL
model = pickle.load(open("model.pkl", "rb"))

# ENCODING MAPS
income_map = {
    "Low": 0,
    "Medium": 1,
    "High": 2
}

satisfaction_map = {
    "Low": 0,
    "Medium": 1,
    "High": 2
}

ads_map = {
    "Low": 0,
    "Medium": 1,
    "High": 2
}

@app.route("/")
def home():

    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    age = int(request.form["age"])

    income = income_map[request.form["income"]]

    rating = float(request.form["rating"])

    satisfaction = satisfaction_map[request.form["satisfaction"]]

    ads = ads_map[request.form["ads"]]

    decision = float(request.form["decision"])

    features = np.array([[
        age,
        income,
        rating,
        satisfaction,
        ads,
        decision
    ]])

    prediction = model.predict(features)[0]

    probability = round(np.max(model.predict_proba(features)) * 100, 2)

    if prediction == 1:
        prediction_text = "Customer Will Purchase"
    else:
        prediction_text = "Customer Will Not Purchase"

    probs = list(model.predict_proba(features)[0] * 100)

    class_names = [
        "Not Purchase",
        "Purchase"
    ]

    return render_template(
        "index.html",
        prediction_text=prediction_text,
        probability=probability,
        probs=probs,
        class_names=class_names
    )

if __name__ == "__main__":

    

    app.run(debug=True, use_reloader=False)