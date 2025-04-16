import yfinance as yf
import pandas as pd
from ta.momentum import RSIIndicator

# Step 1: Download 1 month of daily stock data
ticker = "BHP.AX"
data = yf.download(ticker, period="1mo", interval="1d")

# Step 2: Drop NaNs in 'Close' to avoid RSI issues
# data = data.dropna(subset=["Close"])

# Step 3: Manual RSI calculation
def manual_rsi(data: pd.DataFrame, period: int = 14) -> pd.Series:
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.ewm(span=period, adjust=False).mean()  # EMA instead of SMA
    avg_loss = loss.ewm(span=period, adjust=False).mean()  # EMA instead of SMA

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Step 4: Calculate manual RSI and ta RSI
data['rsi_manual'] = manual_rsi(data)
data['rsi_ta'] = RSIIndicator(close=data['Close'].squeeze(), window=14).rsi()

# Step 5: Print tail to compare
print(data[['Close', 'rsi_manual', 'rsi_ta']].tail())