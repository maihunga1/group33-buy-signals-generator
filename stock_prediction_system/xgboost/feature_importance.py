#feature_importance.py
def generate_shap_rationale(model, X_ticker, features, top_n=3):
    import shap
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_ticker)

    # Ensure we handle classifier output shape
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    shap_summary = sorted(zip(features, shap_values[0]), key=lambda x: abs(x[1]), reverse=True)[:top_n]
    
    interpreted_rationale = []
    for feature, value in shap_summary:
        if feature == "RSI":
            interpreted_rationale.append("RSI is indicating strong momentum" if value > 0 else "RSI shows possible reversal")
        elif feature == "Price_MA_Ratio":
            interpreted_rationale.append("Price is above moving average, bullish trend" if value > 0 else "Price below MA, bearish sign")
        elif feature == "SMA_20":
            interpreted_rationale.append("SMA indicates upward trend" if value > 0 else "SMA suggests slowing trend")
        else:
            interpreted_rationale.append(f"{feature} influences prediction ({value:.4f})")

    return interpreted_rationale
