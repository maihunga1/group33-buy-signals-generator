import pandas as pd
from config import PREDICTION_PROBABILITY_THRESHOLD

def generate_predictions(model_data, data):
    """Generate predictions for latest data using the trained pipeline"""
    latest = data.groupby("Ticker").tail(1).copy()
    
    # Get model components
    model = model_data['model']
    features = model_data['features']
    
    # Prepare features - the pipeline will handle scaling and imputation
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
    
    return latest