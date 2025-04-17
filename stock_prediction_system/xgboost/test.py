import joblib
import pandas as pd
import yfinance as yf
import numpy as np
import ta

# Step 1: Load the trained model from the .pkl file
model_filename = 'data/combined_stock_model.pkl'  # Adjust the path if needed
model = joblib.load(model_filename)

# Step 2: Define the ticker (e.g., 'NAB.AX')
ticker = 'CBA.AX'

# Step 3: Fetch the latest data for the specific ticker (e.g., last day)
stock = yf.Ticker(ticker)
stock_data = stock.history(period="1mo", interval="1d")  # Fetch 1 month of daily data

stock_data['Ticker'] = ticker

# Step 4: Calculate the necessary features for prediction
stock_data['Price_Change'] = stock_data['Close'].pct_change()  # Calculate daily price change

# Calculate the 20-day Simple Moving Average (SMA)
stock_data['SMA_20'] = ta.trend.SMAIndicator(stock_data['Close'], window=20).sma_indicator()

# Calculate RSI
stock_data['RSI'] = ta.momentum.RSIIndicator(stock_data['Close'], window=14).rsi()

# Calculate ATR
stock_data['ATR'] = ta.volatility.AverageTrueRange(
    high=stock_data['High'], 
    low=stock_data['Low'], 
    close=stock_data['Close']
).average_true_range()

# Calculate the Price to Moving Average Ratio
stock_data['Price_MA_Ratio'] = stock_data['Close'] / stock_data['SMA_20']

# Step 5: Prepare the feature set for prediction
features = ['Price_Change', 'SMA_20', 'RSI', 'ATR', 'Price_MA_Ratio']
X_ticker = stock_data[features].dropna().tail(1)  # Get the latest data point for prediction

# Step 6: Make the prediction
prediction = model.predict(X_ticker)

# Step 7: Output the prediction
print(f"Prediction for {ticker}: {'Up' if prediction[0] == 1 else 'Down'}")
