import joblib
import pandas as pd
import matplotlib.pyplot as plt
import shap
import numpy as np

# --- Step 1: Load model and feature names ---
model_data = joblib.load('models/stock_prediction_model.pkl')
model = model_data['model']
features = list(model_data['features'])

# --- Step 2: Extract the RandomForestClassifier (from pipeline or standalone) ---
if hasattr(model, 'named_steps') and 'randomforestclassifier' in model.named_steps:
    rf_model = model.named_steps['randomforestclassifier']
else:
    rf_model = model

# --- Step 3: Prepare a sample dataset to compute SHAP values ---
# Load or reuse the same dataset used to train model if possible
# For demonstration, we'll assume you have your full feature DataFrame
# Replace this with your actual full dataset
data = pd.read_csv("data/processed/processed_data.csv")  # must include the same columns as 'features'
X = data[features]

# --- Step 4: Compute SHAP values ---
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X)

# --- Step 5: Compute mean absolute SHAP values for feature importance ---
if isinstance(shap_values, list) and len(shap_values) == 2:
    shap_vals = np.abs(shap_values[1]).mean(axis=0)
else:
    shap_vals = np.abs(shap_values).mean(axis=0)

# Flatten if needed
shap_vals = shap_vals.flatten()

# Debug lengths
print("✅ SHAP shape:", shap_vals.shape)
print("✅ Feature count:", len(features))

# Ensure length matches
if len(shap_vals) != len(features):
    raise ValueError(f"SHAP values ({len(shap_vals)}) and features ({len(features)}) length mismatch!")

# --- Create summary DataFrame ---
shap_summary_df = pd.DataFrame({
    'Feature': features,
    'SHAP values': shap_vals
}).sort_values(by='SHAP values', ascending=False)



print("\n🔍 SHAP Feature Importances (Mean Absolute):\n")
print(shap_summary_df)

# --- Step 6: Visualize SHAP feature importances ---
plt.figure(figsize=(10, 6))
plt.barh(shap_summary_df['Feature'], shap_summary_df['SHAP values'], color='skyblue')
plt.xlabel('Mean Absolute SHAP Value')
plt.ylabel('Feature')
plt.title('Feature Importance Based on SHAP Values')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
