import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
import joblib
import os

# Ensure model directory exists
os.makedirs('models', exist_ok=True)
os.makedirs('dataset', exist_ok=True)

# Generate an advanced, realistic synthetic dataset
print("Generating advanced synthetic dataset...")
np.random.seed(42)
rows = 20000

# Feature generation
data = pd.DataFrame({
    'amount': np.random.exponential(scale=5000, size=rows) + 100, # mostly Indian amounts in INR
    'time_hour': np.random.randint(0, 24, size=rows),
    'dist_from_home': np.random.exponential(scale=50, size=rows),
    'dist_from_last_tx': np.random.exponential(scale=10, size=rows),
    'ratio_to_median': np.random.lognormal(mean=0, sigma=1, size=rows),
    'used_chip': np.random.choice([0, 1], size=rows, p=[0.3, 0.7]),
    'used_pin': np.random.choice([0, 1], size=rows, p=[0.8, 0.2]),
    'online_order': np.random.choice([0, 1], size=rows, p=[0.4, 0.6]),
})

# Advanced Fraud logic mapping
# 1. High amount, far from home, online order
fraud_1 = (data['amount'] > 25000) & (data['dist_from_home'] > 200) & (data['online_order'] == 1)
# 2. Very high ratio, odd hour, not using pin/chip
fraud_2 = (data['ratio_to_median'] > 8) & ((data['time_hour'] < 5) | (data['time_hour'] > 23)) & (data['used_pin'] == 0)
# 3. Rapid transactions (low dist from last), large dist from home
fraud_3 = (data['dist_from_last_tx'] < 1) & (data['dist_from_home'] > 500) & (data['amount'] > 5000)

data['is_fraud'] = (fraud_1 | fraud_2 | fraud_3).astype(int)

# Inject random noise
noise_idx = np.random.choice(data.index, size=int(rows*0.02), replace=False)
data.loc[noise_idx, 'is_fraud'] = 1 - data.loc[noise_idx, 'is_fraud']

print(f"Dataset generated.\nClass Distribution:\n{data['is_fraud'].value_counts(normalize=True)*100}")

# Save raw dataset
data.to_csv('dataset/advanced_synthetic_transactions.csv', index=False)
print("Saved raw dataset to dataset/advanced_synthetic_transactions.csv")

X = data.drop('is_fraud', axis=1)
y = data['is_fraud']

# Save feature names
feature_names = X.columns.tolist()
joblib.dump(feature_names, 'models/feature_names.pkl')

print("\nScaling features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nBalancing dataset with SMOTE...")
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_scaled, y)
print(f"Resampled shape: {X_res.shape}")

X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

print("\nTraining models...")

models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', max_depth=6, learning_rate=0.1, random_state=42)
}

best_model = None
best_f1 = 0
best_model_name = ""

for name, model in models.items():
    print(f"\nEvaluating {name}...")
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    f1 = f1_score(y_test, preds)
    print(f"Accuracy: {accuracy_score(y_test, preds):.4f} | F1: {f1:.4f}")
    print(classification_report(y_test, preds))
    
    if f1 > best_f1:
        best_f1 = f1
        best_model = model
        best_model_name = name

print(f"\n🏆 Best Model Selected: {best_model_name} (F1: {best_f1:.4f})")

# Feature Importance extraction for explanation
if best_model_name == "XGBoost":
    importances = best_model.feature_importances_
else:
    importances = best_model.feature_importances_

feature_importance_dict = {feat: float(imp) for feat, imp in zip(feature_names, importances)}
# Sort by importance
feature_importance_dict = dict(sorted(feature_importance_dict.items(), key=lambda item: item[1], reverse=True))

joblib.dump(feature_importance_dict, 'models/feature_importances.pkl')

print("\nSaving optimal model artifacts...")
joblib.dump(best_model, 'models/model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')

print("✅ Training complete. Artifacts saved in /models folder.")
