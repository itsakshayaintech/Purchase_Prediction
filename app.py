from flask import Flask, request, render_template
import numpy as np
import joblib

# Load model and scaler
model = joblib.load('model_all_features.pkl')
scaler = joblib.load('scaler_all_features.pkl')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input
        gender = request.form['gender']
        age = float(request.form['age'])
        salary = float(request.form['salary'])

        # Encode gender
        gender_val = 1 if gender == "Male" else 0

        # Prepare input
        input_data = np.array([[gender_val, age, salary]])

        # Scale input
        input_scaled = scaler.transform(input_data)

        # Prediction
        prediction = model.predict(input_scaled)[0]

        # Probability
        probability = model.predict_proba(input_scaled)[0][1] * 100

        # Result
        if prediction == 1:
            result = "User will BUY the product"
        else:
            result = "User will NOT buy the product"

        return render_template(
            'index.html',
            prediction_text=result,
            probability=round(probability, 2)
        )

    except Exception as e:
        return str(e)

if __name__ == "__main__":
    app.run(debug=True)
    
    
    
    