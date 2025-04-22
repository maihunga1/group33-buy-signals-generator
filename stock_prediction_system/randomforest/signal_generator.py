import pandas as pd
import numpy as np
from config import PREDICTION_PROBABILITY_THRESHOLD
from shap_rationale import generate_shap_rationale  # Assuming this function exists

def generate_predictions(model_data, data):
    """Generate predictions for latest data with detailed rationale focusing on top 3 features"""
    latest = data.groupby("Ticker").tail(1).copy()

    # Get model components
    model = model_data['model']
    features = list(model_data['features'])  # ensure it's a list

    # Extract the RandomForestClassifier from the pipeline if needed
    if hasattr(model, 'named_steps') and 'randomforestclassifier' in model.named_steps:
        rf_model = model.named_steps['randomforestclassifier']
    else:
        rf_model = model

    # Generate predictions and probabilities
    predictions = model.predict(latest[features])
    probabilities = model.predict_proba(latest[features])[:, 1]

    latest['Buy_Prediction'] = predictions
    latest['Buy_Probability'] = probabilities
    latest['Recommendation'] = latest['Buy_Probability'].apply(
        lambda x: 'Strong Buy' if x > 0.75 else
                  'Buy' if x > PREDICTION_PROBABILITY_THRESHOLD else
                  'Hold' if x > 0.4 else
                  'Avoid'
    )

    # Create empty columns for results
    latest['Top_3_Features'] = ""
    latest['Rationale'] = ""

    # Process each ticker individually
    for idx, row in latest.iterrows():
        # Create a DataFrame for just this single ticker
        X_row = pd.DataFrame([row[features]], columns=features)
        
        # Generate SHAP values for this specific ticker
        top_features, rationale = generate_shap_rationale(rf_model, X_row, features, top_n=3)
        
        # Format the top features as a string
        top_factor_analysis = [f"{feature}: {shap_val:+.4f}" for feature, shap_val in top_features]
        latest.at[idx, 'Top_3_Features'] = "\n• " + "\n• ".join(top_factor_analysis)
        
        # Store the rationale
        latest.at[idx, 'Rationale'] = "\n".join(rationale)

    # Return the results
    return latest[['Ticker', 'Buy_Prediction', 'Buy_Probability', 'Recommendation', 'Top_3_Features', 'Rationale']]