# model_predict.py
import joblib
import pandas as pd
import yfinance as yf
import numpy as np
from risk_filter import passes_risk_filters
from feature_importance import generate_shap_rationale

model_filename = 'data/combined_stock_model.pkl'
model = joblib.load(model_filename)

top_tickers = ['BHP.AX', 'CBA.AX', 'WES.AX', 'FMG.AX', 'TLS.AX']

def fetch_stock_data(ticker):
    stock = yf.Ticker(ticker)
    return stock.history(period="1mo", interval="1d")

def prepare_features(stock_data):
    stock_data['Price_Change'] = stock_data['Close'].pct_change()
    stock_data['SMA_20'] = stock_data['Close'].rolling(window=20).mean()
    stock_data['EMA_12'] = stock_data['Close'].ewm(span=12).mean()
    stock_data['EMA_26'] = stock_data['Close'].ewm(span=26).mean()
    stock_data['MACD'] = stock_data['EMA_12'] - stock_data['EMA_26']
    stock_data['MACD_Signal'] = stock_data['MACD'].ewm(span=9).mean()
    stock_data['MACD_Diff'] = stock_data['MACD'] - stock_data['MACD_Signal']
    
    stock_data['RSI'] = stock_data['Close'].pct_change().rolling(window=14).mean()
    stock_data['Stoch'] = ((stock_data['Close'] - stock_data['Low'].rolling(window=14).min()) /
                           (stock_data['High'].rolling(window=14).max() - stock_data['Low'].rolling(window=14).min()))
    stock_data['Stoch_Signal'] = stock_data['Stoch'].rolling(window=3).mean()

    stock_data['ATR'] = (stock_data['High'] - stock_data['Low']).rolling(window=14).mean()
    stock_data['Price_MA_Ratio'] = stock_data['Close'] / stock_data['SMA_20']
    stock_data['Volume_Change'] = stock_data['Volume'] / stock_data['Volume'].rolling(window=5).mean()
    stock_data['OBV'] = (np.sign(stock_data['Close'].diff()) * stock_data['Volume']).fillna(0).cumsum()
    stock_data['Return_5D'] = stock_data['Close'].pct_change(periods=5)
    stock_data['Return_20D'] = stock_data['Close'].pct_change(periods=20)
    stock_data['BB_Width'] = ((stock_data['Close'].rolling(20).mean() + 2 * stock_data['Close'].rolling(20).std()) -
                              (stock_data['Close'].rolling(20).mean() - 2 * stock_data['Close'].rolling(20).std())) / stock_data['Close']
    stock_data['MA_Crossover_12_26'] = (stock_data['EMA_12'] > stock_data['EMA_26']).astype(int)
    return stock_data

dashboard_data = []

for ticker in top_tickers:
    stock_data = fetch_stock_data(ticker)
    stock_data = prepare_features(stock_data)

    features = [
        'Price_Change', 'SMA_20', 'RSI', 'ATR', 'Price_MA_Ratio',
        'MACD', 'MACD_Signal', 'MACD_Diff',
        'Stoch', 'Stoch_Signal',
        'Volume_Change', 'OBV',
        'Return_5D', 'Return_20D',
        'BB_Width', 'MA_Crossover_12_26'
    ]
    X_ticker = stock_data[features].dropna().tail(1)

    prediction = model.predict(X_ticker)[0]
    prediction_label = "Yes" if prediction == 1 else "No"

    if not passes_risk_filters(stock_data):
        print(f"Skipping {ticker} due to risk filter")
        continue

    shap_rationale = generate_shap_rationale(model, X_ticker, X_ticker.columns)

    # Top 3 weights for feature importance
    top_features = model.feature_importances_
    importance = sorted(zip(X_ticker.columns, top_features), key=lambda x: x[1], reverse=True)[:3]

    dashboard_data.append({
        'ticker': ticker,
        'buy_signal': prediction_label,
        'rationale': shap_rationale[0],  # Most important interpreted feature
        'feature_importance_weights': [
            {'feature': f, 'weight': round(w, 2)} for f, w in importance
        ]
    })

for stock in dashboard_data:
    print(f"Ticker: {stock['ticker']}")
    print(f"Buy Signal: {stock['buy_signal']}")
    print(f"Rationale: {stock['rationale']}")
    print("Feature Importance:")
    for item in stock['feature_importance_weights']:
        print(f"  {item['feature']}: {item['weight']}")
    print("-" * 40)
