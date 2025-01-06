from flask import Flask, request, jsonify
import pickle
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load the trained Logistic Regression model
model_filename = 'logistic_regression_model.pkl'
with open(model_filename, 'rb') as f:
    model = pickle.load(f)

# Initialize the scaler used during training
scaler = StandardScaler()

@app.route('/')
def home():
    return "Welcome to the Churn Prediction API!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the POST request
        data = request.get_json()

        # Extract features from the request data
        features = np.array(data['features']).reshape(1, -1)
        
        # Scale the features using the same scaler as during training
        features_scaled = scaler.fit_transform(features)
        
        # Get the prediction from the model
        prediction = model.predict(features_scaled)

        # Return prediction as JSON response
        result = {"prediction": int(prediction[0])}
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)



