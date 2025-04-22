import shap
import numpy as np
import pandas as pd
from typing import List, Tuple

def generate_shap_rationale(model, X_ticker, features, top_n=3):
    """
    Generates SHAP explanations with bullet-point rationale in exact requested format.
    
    Parameters:
        model: Trained sklearn model/pipeline
        X_ticker: Single observation (DataFrame or array)
        features: List of feature names
        top_n: Number of top features to explain
        
    Returns:
        Tuple of (top_features, rationale) where:
        - top_features: List of (feature_name, shap_value)
        - rationale: List of strings in format ["🔹 Feature: Interpretation", ...]
    """
    # ======================
    # 1. Input Validation
    # ======================
    if not hasattr(model, 'predict_proba'):
        raise ValueError("Model must implement predict_proba()")
        
    # Convert to DataFrame if needed
    if isinstance(X_ticker, np.ndarray):
        if X_ticker.ndim != 2 or X_ticker.shape[0] != 1:
            raise ValueError("Input must be single observation")
        X_ticker = pd.DataFrame(X_ticker, columns=features)
    elif isinstance(X_ticker, pd.DataFrame):
        X_ticker = X_ticker.iloc[[0]]  # Ensure single row
    else:
        raise TypeError("Input must be DataFrame or numpy array")

    # ======================
    # 2. Model Extraction
    # ======================
    if hasattr(model, 'named_steps'):
        for step in model.named_steps.values():
            if hasattr(step, 'predict_proba'):
                model = step
                break

    # ======================
    # 3. SHAP Calculation
    # ======================
    explainer = shap.TreeExplainer(model)
    try:
        shap_values = explainer.shap_values(X_ticker)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Use positive class
        shap_values = shap_values[0]  # Get first (only) observation
    except Exception as e:
        raise ValueError(f"SHAP calculation failed: {str(e)}")

    # ======================
    # 4. Feature Processing
    # ======================
    def get_scalar(value):
        """Safely extract single value from array-like"""
        if isinstance(value, (np.ndarray, pd.Series)):
            return value.item() if value.size == 1 else value[0]
        return value

    try:
        # Extract all values as scalars
        row = {f: get_scalar(X_ticker[f].values[0]) for f in features}
        shap_values = [get_scalar(v) for v in shap_values]
    except Exception as e:
        raise ValueError(f"Feature extraction failed: {str(e)}")

    # ======================
    # 5. Generate Rationale
    # ======================
    top_features = sorted(zip(features, shap_values), 
                         key=lambda x: abs(x[1]), reverse=True)[:top_n]
    
    rationale = []
    for feature, _ in top_features:
        val = row[feature]
        
        # Enhanced technical indicator interpretations
        if feature == "RSI":
            if val > 70:
                interpretation = "Overbought (RSI > 70) - Potential overbought condition"
            elif val < 30:
                interpretation = "Oversold (RSI < 30) - Potential oversold condition"
            else:
                interpretation = f"Neutral ({val:.1f}) - No extreme conditions"
                
        elif feature == "MACD":
            interpretation = "Bullish crossover" if val > 0 else "Bearish crossover"
            interpretation += f" ({val:.4f})"
            
        elif feature == "Stoch" or feature == "Stoch_Signal":
            if val > 80:
                interpretation = f"Overbought ({val:.1f}) - Potential reversal signal"
            elif val < 20:
                interpretation = f"Oversold ({val:.1f}) - Potential bounce signal"
            else:
                interpretation = f"Neutral ({val:.1f}) - No extreme signals"
                
        elif feature == "OBV":
            if val > 1e9:  # Billions
                vol = f"{val/1e9:.2f}B"
                trend = "Strong accumulation"
            elif val > 1e6:  # Millions
                vol = f"{val/1e6:.2f}M"
                trend = "Moderate accumulation"
            else:
                vol = f"{val:,.0f}"
                trend = "Neutral volume"
            interpretation = f"{vol} - {trend}"
            
        elif feature == "BB_Width":
            if val > 0.15:
                interpretation = f"Wide ({val:.4f}) - High volatility expected"
            elif val < 0.05:
                interpretation = f"Narrow ({val:.4f}) - Low volatility expected"
            else:
                interpretation = f"Normal ({val:.4f}) - Typical volatility"
                
        elif feature in ["SMA_20", "EMA_12", "EMA_26"]:
            price = row.get('Close', None)
            if price:
                relation = "above" if price > val else "below"
                interpretation = f"{val:.2f} (Price is {relation})"
            else:
                interpretation = f"{val:.2f}"
                
        elif feature == "Price_MA_Ratio":
            if val > 1.05:
                interpretation = "Significantly above MA - Bullish"
            elif val < 0.95:
                interpretation = "Significantly below MA - Bearish"
            else:
                interpretation = "Near MA - Neutral"
            interpretation += f" ({val:.2f})"
            
        elif feature == "ATR":
            if val > 0.1:
                interpretation = f"High volatility ({val:.4f})"
            else:
                interpretation = f"Normal volatility ({val:.4f})"
                
        elif feature == "Volume_Change":
            if val > 0.5:
                interpretation = f"Significant volume spike (+{val:.1%})"
            elif val < -0.5:
                interpretation = f"Significant volume drop ({val:.1%})"
            else:
                interpretation = f"Normal volume change ({val:.1%})"
                
        elif feature == "MA_Crossover_12_26":
            interpretation = "Bullish crossover detected" if val == 1 else "No crossover"
            
        else:
            interpretation = f"Value: {val:.4f}"
        
        rationale.append(f"🔹 {feature}: {interpretation}")
    
    return top_features, rationale