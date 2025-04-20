import joblib
import pandas as pd
import yfinance as yf
import numpy as np
import ta
import shap

# Step 1: Load the trained model from the .pkl file
model_filename = 'data/combined_stock_model.pkl'  # Adjust the path if needed
model = joblib.load(model_filename)

# Define the ticker(s) for the top stocks
top_tickers = ['CBA.AX', 'BHP.AX', 'NAB.AX']  # Example top 3 stock tickers

# Features you are using in the model
features = ['Price_Change', 'SMA_20', 'RSI', 'ATR', 'Price_MA_Ratio']

# Function to fetch and prepare stock data (as done earlier)
def fetch_and_prepare_data(ticker):
    stock = yf.Ticker(ticker)
    stock_data = stock.history(period="1mo", interval="1d")  # Fetch 1 month of daily data

    stock_data['Price_Change'] = stock_data['Close'].pct_change()  # Calculate daily price change
    stock_data['SMA_20'] = ta.trend.SMAIndicator(stock_data['Close'], window=20).sma_indicator()  # 20-day SMA
    stock_data['RSI'] = ta.momentum.RSIIndicator(stock_data['Close'], window=14).rsi()  # RSI
    stock_data['ATR'] = ta.volatility.AverageTrueRange(
        high=stock_data['High'], low=stock_data['Low'], close=stock_data['Close']
    ).average_true_range()  # ATR
    stock_data['Price_MA_Ratio'] = stock_data['Close'] / stock_data['SMA_20']  # Price to SMA Ratio

    return stock_data[features].dropna().tail(1)  # Prepare features for the latest data point

# Step 2: Get predictions and SHAP values for each stock in the top predictions list
explainers = {}
shap_values_dict = {}

for ticker in top_tickers:
    # Fetch and prepare the data
    X_ticker = fetch_and_prepare_data(ticker)

    # Step 3: Make the prediction for each stock
    prediction = model.predict(X_ticker)

    # Step 4: Get the SHAP explanation
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_ticker)

    explainers[ticker] = explainer
    shap_values_dict[ticker] = shap_values

    # Print prediction for the stock
    print(f"Prediction for {ticker}: {'Up' if prediction[0] == 1 else 'Down'}")

    # Step 5: Get Feature Importance from the model and SHAP values
    # Model-based feature importance (Global importance)
    feature_importance = model.feature_importances_

    # Print the feature importance for the stock prediction
    print("Model Feature Importance:")
    for feature, importance in zip(features, feature_importance):
        print(f"{feature}: {importance:.4f}")

    # Top contributing features using SHAP
    print("\nSHAP Top Contributing Features for the Prediction:")
    top_contributing_features = sorted(zip(features, shap_values[0]), key=lambda x: abs(x[1]), reverse=True)[:3]
    for feature, value in top_contributing_features:
        print(f"{feature}: {value:.4f}")

    # Step 6: Domain-Based Rationale for the Stock's Prediction
    rationale = []

    # Example rationale based on technical indicators
    if X_ticker['RSI'].values[0] > 70:
        rationale.append("RSI is high, indicating bullish momentum.")
    elif X_ticker['RSI'].values[0] < 30:
        rationale.append("RSI is low, indicating bearish momentum.")

    if X_ticker['Price_MA_Ratio'].values[0] > 1.05:
        rationale.append("Price is above the 20-day moving average, suggesting upward momentum.")
    elif X_ticker['Price_MA_Ratio'].values[0] < 0.95:
        rationale.append("Price is below the 20-day moving average, suggesting downward pressure.")

    # Combine all the rationale points
    print("\nRationale for the prediction:")
    for point in rationale:
        print(f"- {point}")
    
    print("\n==============================\n")

