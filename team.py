from flask import Flask, request, jsonify
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder  # Add this import

app = Flask(__name__)

# Load models and scaler
linear_model = joblib.load('linear_regression_model.pkl')
rf_model = joblib.load('random_forest_model.pkl')
svm_model = joblib.load('svm_model.pkl')
xgb_model = joblib.load('xgboost_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    # Preprocess input data
    input_features = pd.DataFrame(data, index=[0])
    columns_to_drop = ['Name', 'Image', 'Name_URL', 'ID', 'Players', 'Passing1', 'Positioning2', 'Ligue', 'saison']
    columns_to_drop = [col for col in columns_to_drop if col in input_features.columns]
    input_features = input_features.drop(columns=columns_to_drop)
    # input_features = input_features.drop(columns=['Name', 'Image', 'Name_URL', 'ID', 'Players', 'Passing1', 'Positioning2', 'Ligue', 'saison'])
    for col in ['Speed', 'Dribbling', 'Passing', 'Positioning', 'Crossing', 'Shooting', 'Aggression', 'Pressure', 'Team_width', 'Defender_line']:
        input_features[col] = LabelEncoder().fit_transform(input_features[col])
    input_features_scaled = scaler.transform(input_features)

    # Predict using models
    linear_pred = linear_model.predict(input_features_scaled)[0]
    rf_pred = rf_model.predict(input_features_scaled)[0]
    svm_pred = svm_model.predict(input_features_scaled)[0]
    xgb_pred = xgb_model.predict(input_features_scaled)[0]

    return jsonify({
        'Linear Regression': linear_pred,
        'Random Forest': rf_pred,
        'SVM': svm_pred,
        'XGBoost': xgb_pred
    })

if __name__ == '__main__':
    app.run(debug=True)
