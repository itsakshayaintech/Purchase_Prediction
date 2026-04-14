from flask import Flask, request, render_template
import numpy as np

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        gender = request.form['gender']
        age = float(request.form['age'])
        salary = float(request.form['salary'])

        # SIMPLE TEST LOGIC (to check if UI works)
        if age > 40 and salary > 50000:
            prediction = 1
        else:
            prediction = 0

        if prediction == 1:
            result = "User will BUY the product"
        else:
            result = "User will NOT buy the product"

        return render_template('index.html', prediction_text=result)

    except Exception as e:
        return str(e)

if __name__ == "__main__":
    app.run(debug=True)
    
    
    
    