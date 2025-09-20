from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load('backend/credit_fraud.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    # Assuming the input data is a dictionary of features
    features = pd.DataFrame([data])
    prediction = model.predict(features)
    prediction_proba = model.predict_proba(features)
    result = {
        'prediction': int(prediction[0]),
        'probability': prediction_proba[0].tolist()
    }
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
