import os
import traceback
import joblib
import numpy as np
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Paths
MODEL_PATH = 'models/model.pkl'
SCALER_PATH = 'models/scaler.pkl'
FEATURES_PATH = 'models/feature_names.pkl'
IMPORTANCE_PATH = 'models/feature_importances.pkl'

model = None
scaler = None
feature_names = None
feature_importances = None

def load_ml_objects():
    global model, scaler, feature_names, feature_importances
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH) and os.path.exists(FEATURES_PATH):
        try:
            model = joblib.load(MODEL_PATH)
            scaler = joblib.load(SCALER_PATH)
            feature_names = joblib.load(FEATURES_PATH)
            if os.path.exists(IMPORTANCE_PATH):
                feature_importances = joblib.load(IMPORTANCE_PATH)
            print("✅ AI Models & Scalers loaded successfully.")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
    else:
        print("⚠️ Model files not found. Run train_model.py first.")

load_ml_objects()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about_contact():
    return render_template('about_contact.html')

@app.route('/api/status', methods=['GET'])
def get_status():
    if model and scaler:
        return jsonify({"status": "online", "model": type(model).__name__, "features": feature_names}), 200
    return jsonify({"status": "offline"}), 503

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or scaler is None:
        load_ml_objects()
        if model is None:
            return jsonify({"error": "AI Engine offline. Please run the training script."}), 503

    try:
        data = request.json
        
        # Extract advanced features
        features_dict = {
            'amount': float(data.get('amount', 0)),
            'time_hour': float(data.get('time_hour', 0)),
            'dist_from_home': float(data.get('dist_from_home', 0)),
            'dist_from_last_tx': float(data.get('dist_from_last_tx', 0)),
            'ratio_to_median': float(data.get('ratio_to_median', 1)),
            'used_chip': int(data.get('used_chip', 0)),
            'used_pin': int(data.get('used_pin', 0)),
            'online_order': int(data.get('online_order', 0))
        }

        # Ensure correct order
        input_array = np.array([[features_dict[f] for f in feature_names]])
        
        # Scale
        scaled_features = scaler.transform(input_array)
        
        # Predict
        prediction_num = model.predict(scaled_features)[0]
        prediction_label = "Fraud" if prediction_num == 1 else "Legit"
        
        # Probability
        probs = model.predict_proba(scaled_features)[0]
        probability = float(probs[1]) if len(probs) > 1 else float(prediction_num)
        
        # Risk Classification
        if probability < 0.35:
            risk_level = "Low"
        elif probability < 0.70:
            risk_level = "Medium"
        else:
            risk_level = "High"

        # Generate Explainability (Top contributing factors for this specific prediction simply based on global importance * input scale impact)
        # For true SHAP values, we'd need heavy compute, so we use a heuristic based on feature deviations from mean
        explanations = []
        if feature_importances:
            mean_vals = scaler.mean_
            std_vals = scaler.scale_
            
            deviations = []
            for i, feat in enumerate(feature_names):
                val = input_array[0][i]
                z_score = abs(val - mean_vals[i]) / (std_vals[i] + 1e-6)
                global_imp = feature_importances.get(feat, 0)
                score = z_score * global_imp
                deviations.append((feat, score, val))
            
            deviations.sort(key=lambda x: x[1], reverse=True)
            top_factors = deviations[:3]
            for f, s, v in top_factors:
                if f == 'amount': explanations.append(f"Amount (₹{v:.2f}) is unusual.")
                elif f == 'dist_from_home': explanations.append(f"Distance from home ({v:.1f}km) is high.")
                elif f == 'ratio_to_median': explanations.append(f"Purchase ratio ({v:.1f}x) is abnormal.")
                elif f == 'online_order' and v == 1: explanations.append("Online orders carry higher baseline risk.")
                elif f == 'used_pin' and v == 0: explanations.append("Transaction processed without a PIN.")
                else: explanations.append(f"Anomaly detected in {f}.")

        if not explanations:
            explanations = ["Standard transaction pattern matched.", "No severe anomalies found."]

        return jsonify({
            "prediction": prediction_label,
            "probability": probability,
            "risk_level": risk_level,
            "explanations": list(set(explanations))[:3] # unique top 3
        })
        
    except Exception as e:
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
