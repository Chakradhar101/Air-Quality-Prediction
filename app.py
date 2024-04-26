from flask import Flask, render_template, request
import joblib
import numpy as np
import os
app = Flask(__name__)

# Load the saved Random Forest Regressor model
model = joblib.load('random_forest_model.joblib')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get user input from the form
        soi = float(request.form['soi'])
        noi = float(request.form['noi'])
        rpi = float(request.form['rpi'])
        spmi = float(request.form['spmi'])

        # Make predictions using the loaded model
        input_data = np.array([[soi, noi, rpi, spmi]])
        predicted_value = model.predict(input_data)[0]

        # Classify air quality based on the predicted value
        air_quality = classify_air_quality(predicted_value)

        return render_template('result.html', predicted_value=predicted_value, air_quality=air_quality)

def classify_air_quality(value):
    if value <= 50:
        return "Good"
    elif 50 < value <= 100:
        return "Moderate"
    elif 100 < value <= 200:
        return "Poor"
    elif 200 < value <= 300:
        return "Unhealthy"
    elif 300 < value <= 400:
        return "Very unhealthy"
    elif value > 400:
        return "Hazardous"

if __name__ == '__main__':
    app.run(debug=True)
