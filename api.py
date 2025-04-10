from flask import Flask, request, jsonify
import numpy as np
import joblib
import os
from warnings import simplefilter # For warning control
simplefilter('ignore')#ignore warning to keep output clean

app = Flask(__name__)
# load model and feature selector
model = joblib.load('rfc_model.pkl')
selector = joblib.load('feature_selector.pkl')

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Loan outcome prediction API is up."})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Expected features
        required_fields = ["age", "cash_incoming_30days",
        "latitude", "longitude",
        "accuracy", "Day",
        "Month", "Year",
        "Day_of_Week"]

        # Check for missing fields
        if all(field in data for field in required_fields):
            # Format input
            input_array = np.array([[data[field] for field in required_fields]])
            input_selected = selector.transform(input_array)

            # Make prediction and probability
            prediction = model.predict(input_selected)[0]
            probability = model.predict_proba(input_selected)[0][1]
            
            result = "repaid" if prediction == 1 else "defaulted"
            return jsonify({"prediction" : result, "probability_repaid":round(probability, 4 )})
        else:
            return jsonify({"error": "Missing one or more required fields"}), 400
    
    except Exception as e:
        return jsonify({"error":str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)