import shap
import numpy as np

def generate_shap_rationale(model, X_ticker, features, top_n=3):
    """
    Generate SHAP-based explanations and rationale for a single prediction input.
    Returns both:
      - top_features: List of (feature_name, shap_value)
      - rationale: List of human-readable explanations
    """

    # If model is inside pipeline, extract actual estimator
    if hasattr(model, 'named_steps'):
        for name, step in model.named_steps.items():
            if hasattr(step, 'predict_proba') and hasattr(step, 'feature_importances_'):
                model = step
                break
        else:
            raise ValueError("❌ No suitable estimator found inside pipeline.")
    
    # Use SHAP TreeExplainer for Random Forest
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_ticker)

    # Support binary classification: use class 1 (positive)
    shap_values_instance = shap_values[1][0] if isinstance(shap_values, list) else shap_values[0][0]

    # Get top-N features by absolute SHAP value
    top_features = sorted(zip(features, shap_values_instance), key=lambda x: abs(x[1]), reverse=True)[:top_n]

    # Fetch feature values from X_ticker
    row = X_ticker.iloc[0]
    rationale = []

    for feature, shap_value in top_features:
        val = row[feature]
        explanation = f"🔹 {feature}: "

        if feature == "RSI":
            if val > 70:
                explanation += "Overbought (RSI > 70)"
            elif val < 30:
                explanation += "Oversold (RSI < 30)"
            else:
                explanation += f"Neutral at {val:.1f}"

        elif feature == "MACD":
            explanation += "Positive trend" if val > 0 else "Negative trend"
        elif feature == "MACD_Diff":
            explanation += "Bullish divergence" if val > 0 else "Bearish signal"
        elif feature == "MACD_Signal":
            explanation += f"MACD signal = {val:.4f}"

        elif feature == "Stoch":
            explanation += (
                "Overbought (>80)" if val > 80 else
                "Oversold (<20)" if val < 20 else
                f"Neutral at {val:.1f}"
            )
        elif feature == "Stoch_Signal":
            explanation += f"Stochastic Signal = {val:.1f}"

        elif feature in ["SMA_20", "EMA_12", "EMA_26"]:
            explanation += f"MA = {val:.2f}"

        elif feature == "Price_MA_Ratio":
            explanation += (
                "Above MA" if val > 1.05 else
                "Below MA" if val < 0.95 else
                "Near MA"
            ) + f" ({val:.2f})"

        elif feature == "ATR":
            explanation += f"ATR = {val:.4f} (Volatility)"
        elif feature == "BB_Width":
            explanation += f"Band width = {val:.4f}"

        elif feature in ["Price_Change", "Return_5D", "Return_20D"]:
            explanation += "Upward momentum" if val > 0 else "Downward momentum"

        elif feature == "Volume_Change":
            explanation += f"Volume spike {val:.1%}" if val > 1 else f"Volume drop {val:.1%}"
        elif feature == "OBV":
            explanation += f"OBV = {val:.0f}"

        elif feature == "MA_Crossover_12_26":
            explanation += "Bullish crossover" if val == 1 else "No crossover"

        else:
            explanation += f"{val:.4f}"

        rationale.append(explanation)

    return top_features, rationale
