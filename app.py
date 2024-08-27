from flask import Flask, request, render_template, jsonify
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
import numpy as np
import os

app = Flask(__name__)

# Get the absolute path for the model files
current_directory = os.path.dirname(__file__)
rf_model_path = os.path.join(current_directory, r'D:\diabetics_prediction\models\best_rf_pipeline (1).pkl')
nn_model_path = os.path.join(current_directory, r'D:\diabetics_prediction\models\nn_model.keras')

# Load the trained models using absolute paths
if os.path.exists(rf_model_path):
    rf_model = joblib.load(rf_model_path)
    print("Random Forest model loaded successfully.")
else:
    raise FileNotFoundError(f"Random Forest model file not found at {rf_model_path}")

if os.path.exists(nn_model_path):
    nn_model = load_model(nn_model_path)
    # Explicitly set steps_per_execution if it's None
    if nn_model.steps_per_execution is None:
        nn_model.steps_per_execution = 1
    print("Neural Network model loaded successfully.")
else:
    raise FileNotFoundError(f"Neural Network model file not found at {nn_model_path}")

# Home route to serve the form
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    data = request.form.to_dict()
    input_data = pd.DataFrame([data])

    # Preprocess data for Random Forest model
    rf_predictions = rf_model.predict(input_data)
    rf_predictions_rounded = np.round(rf_predictions)

    # Preprocess data for Neural Network model
    preprocessed_data = rf_model.named_steps['preprocessor'].transform(input_data)
    if isinstance(preprocessed_data, np.ndarray) is False:
        preprocessed_data = preprocessed_data.toarray()
    
    nn_predictions = nn_model.predict(preprocessed_data)
    nn_predictions_rounded = np.round(nn_predictions)

    # Combined prediction
    combined_prediction = np.round((rf_predictions_rounded + nn_predictions_rounded) / 2)

    # Render the result page with predictions
    return render_template('result.html',
                           rf_prediction=int(rf_predictions_rounded[0]),
                           nn_prediction=int(nn_predictions_rounded[0]),
                           combined_prediction=int(combined_prediction[0]))

if __name__ == "__main__":
    app.run(debug=True)
