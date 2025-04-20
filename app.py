# app.py
from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load model and scaler
model = joblib.load('titanic_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    pclass = int(request.form['Pclass'])
    sex = 1 if request.form['Sex'] == 'male' else 0
    age = float(request.form['Age'])
    fare = float(request.form['Fare'])
    family_size = int(request.form['FamilySize'])

    input_data = np.array([[pclass, sex, age, fare, family_size]])
    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]
    result = 'Survived' if prediction == 1 else 'Did not Survive'

    return render_template('index.html', prediction_text=f"The passenger would have: {result}")

if __name__ == "__main__":
    app.run(debug=True)
