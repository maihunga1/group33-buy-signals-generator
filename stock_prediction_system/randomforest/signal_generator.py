import pandas as pd
import numpy as np
from config import PREDICTION_PROBABILITY_THRESHOLD

def generate_predictions(model_data, data):
    """Generate predictions for latest data with detailed rationale focusing on top 3 features"""
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
    
    # Extract feature importance from the model
    # Note: For pipeline models, we need to access the actual classifier
    if hasattr(model, 'named_steps') and 'randomforestclassifier' in model.named_steps:
        classifier = model.named_steps['randomforestclassifier']
        feature_importance = classifier.feature_importances_
    else:
        # For simple models without pipeline
        feature_importance = getattr(model, 'feature_importances_', None)
    
    # Generate rationale based on top 3 important features
    latest['Rationale'] = ""
    for idx, row in latest.iterrows():
        ticker = row['Ticker']
        recommendation = row['Recommendation']
        probability = row['Buy_Probability']
        
        # Start with confidence level
        confidence_text = f"{probability:.1%} confidence - "
        if recommendation == 'Strong Buy':
            rationale_intro = f"{confidence_text}Most influential factors:"
        elif recommendation == 'Buy':
            rationale_intro = f"{confidence_text}Key factors supporting this decision:"
        elif recommendation == 'Hold':
            rationale_intro = f"{confidence_text}Mixed signals with these main factors:"
        else:
            rationale_intro = f"{confidence_text}Main concerns include:"
        
        top_factor_analysis = []
        
        # Add key technical indicators commentary based on top 3 features
        if feature_importance is not None:
            # Get indices of top 3 most important features
            top_feature_indices = np.argsort(feature_importance)[-3:][::-1]
            
            for i, feature_idx in enumerate(top_feature_indices):
                feature = features[feature_idx]
                importance = feature_importance[feature_idx]
                value = row[feature]
                
                # Format the feature importance as a percentage
                importance_pct = f"{importance:.1%}"
                
                # Add feature number and importance percentage
                factor_text = f"#{i+1} {feature} (importance: {importance_pct}): "
                
                # RSI analysis
                if feature == 'RSI':
                    if value > 70:
                        factor_text += f"Overbought at {value:.1f}"
                    elif value < 30:
                        factor_text += f"Oversold at {value:.1f}"
                    else:
                        factor_text += f"Neutral at {value:.1f}"
                
                # MACD analysis
                elif feature in ['MACD', 'MACD_Signal', 'MACD_Diff']:
                    if feature == 'MACD_Diff':
                        direction = "Bullish" if value > 0 else "Bearish"
                        factor_text += f"{direction} momentum ({value:.4f})"
                    elif feature == 'MACD':
                        direction = "Positive" if value > 0 else "Negative"
                        factor_text += f"{direction} trend ({value:.4f})"
                    else:
                        factor_text += f"Signal line at {value:.4f}"
                
                # Moving average analysis
                elif feature in ['SMA_20', 'EMA_12', 'EMA_26']:
                    current_price = row['Close']
                    relation = "above" if current_price > value else "below"
                    ma_type = feature.split('_')[0]  # SMA or EMA
                    period = feature.split('_')[1]   # 20, 12, 26
                    factor_text += f"Price {relation} {ma_type}-{period} by {(current_price/value - 1):.1%}"
                
                # Bollinger Bands analysis
                elif feature == 'BB_Width':
                    if value > 0.05:
                        factor_text += f"High volatility ({value:.2f})"
                    else:
                        factor_text += f"Low volatility ({value:.2f})"
                
                # Stochastic oscillator
                elif feature in ['Stoch', 'Stoch_Signal']:
                    if value > 80:
                        factor_text += f"Overbought at {value:.1f}"
                    elif value < 20:
                        factor_text += f"Oversold at {value:.1f}"
                    else:
                        factor_text += f"Neutral at {value:.1f}"
                
                # Volume indicators
                elif feature == 'OBV':
                    trend = "Increasing" if value > row['OBV'] * 0.95 else "Decreasing"
                    factor_text += f"{trend} volume trend"
                
                # Price momentum
                elif feature == 'Price_Change':
                    direction = "Upward" if value > 0 else "Downward"
                    factor_text += f"{direction} momentum ({value:.1%})"
                
                # Price to moving average ratio
                elif feature == 'Price_MA_Ratio':
                    if value > 1.05:
                        factor_text += f"Significantly above MA ({value:.2f})"
                    elif value < 0.95:
                        factor_text += f"Significantly below MA ({value:.2f})"
                    else:
                        factor_text += f"Near MA ({value:.2f})"
                        
                # Average True Range
                elif feature == 'ATR':
                    factor_text += f"Volatility indicator at {value:.4f}"
                
                # Other indicators with simple value reporting
                else:
                    factor_text += f"{value:.4f}"
                
                top_factor_analysis.append(factor_text)
        
        # Combine rationale intro with top factor analysis
        latest.at[idx, 'Rationale'] = rationale_intro + "\n• " + "\n• ".join(top_factor_analysis)
    
    return latest