from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

app = Flask(__name__)

# Load the trained scaler and models
scaler = joblib.load('scaler.pkl')
linear_model = joblib.load('linear_regression_model.pkl')
rf_model = joblib.load('random_forest_model.pkl')
svm_model = joblib.load('svm_model.pkl')
xgb_model = joblib.load('xgboost_model.pkl')

# Define function to preprocess input data
def preprocess_input(input_data):
    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_data])

    # Convert categorical variables to numerical using LabelEncoder
    categorical_cols = ['Speed', 'Dribbling', 'Passing', 'Positioning', 'Crossing', 'Shooting', 
                        'Aggression', 'Pressure', 'Team_width', 'Defender_line']
    for col in categorical_cols:
        input_df[col] = LabelEncoder().fit_transform(input_df[col])

    # Return preprocessed input
    return input_df

@app.route('/predict', methods=['POST'])
def predict():
    # Get input data from request
    input_data = request.json
    
    # Preprocess input data
    input_df = preprocess_input(input_data)

    # Scale input features
    input_features_scaled = scaler.transform(input_df)

    # Make predictions using each model
    linear_pred = linear_model.predict(input_features_scaled)
    rf_pred = rf_model.predict(input_features_scaled)
    svm_pred = svm_model.predict(input_features_scaled)
    xgb_pred = xgb_model.predict(input_features_scaled)

    # Return predictions
    return jsonify({
        "linear_regression_prediction": float(linear_pred),
        "random_forest_prediction": float(rf_pred),
        "svm_prediction": float(svm_pred),
        "xgboost_prediction": float(xgb_pred)
    })

if __name__ == '__main__':
    app.run(debug=True)
