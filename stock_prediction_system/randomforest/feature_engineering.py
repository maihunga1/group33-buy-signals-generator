import pandas as pd
import ta
from config import RETURN_THRESHOLD
from utils import get_data_path

def create_features(df):
    """Create technical indicators and features"""
    if df.empty:
        raise ValueError("Empty DataFrame received - cannot create features")
        
    # Trend indicators
    df['SMA_20'] = ta.trend.SMAIndicator(df['Close'], window=20).sma_indicator()
    df['EMA_12'] = ta.trend.EMAIndicator(df['Close'], window=12).ema_indicator()
    df['EMA_26'] = ta.trend.EMAIndicator(df['Close'], window=26).ema_indicator()
    df['MACD'] = ta.trend.MACD(df['Close']).macd()
    df['MACD_Signal'] = ta.trend.MACD(df['Close']).macd_signal()
    df['MACD_Diff'] = ta.trend.MACD(df['Close']).macd_diff()
    
    # Momentum indicators
    df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
    df['Stoch'] = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close']).stoch()
    df['Stoch_Signal'] = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close']).stoch_signal()
    
    # Volatility indicators
    df['ATR'] = ta.volatility.AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close']).average_true_range()
    df['BB_High'] = ta.volatility.BollingerBands(df['Close']).bollinger_hband()
    df['BB_Low'] = ta.volatility.BollingerBands(df['Close']).bollinger_lband()
    df['BB_Width'] = (df['BB_High'] - df['BB_Low']) / df['Close']
    
    # Volume indicators
    df['OBV'] = ta.volume.OnBalanceVolumeIndicator(df['Close'], df['Volume']).on_balance_volume()
    
    # Price-based features
    df['Price_Change'] = df['Close'].pct_change()
    df['Price_MA_Ratio'] = df['Close'] / df['SMA_20']
    
    # Target creation - forward returns
    df['Return_1w'] = df['Close'].shift(-1) / df['Close'] - 1
    df['Buy_Signal'] = (df['Return_1w'] > RETURN_THRESHOLD).astype(int)

    processed_data_path = get_data_path("processed_data.csv", subfolder='processed')
    df.to_csv(processed_data_path, index=False)
    print(f"💾 Saved processed data to {processed_data_path}")
    
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print("⚠️ Warning: Data contains missing values")
        print(missing[missing > 0])
    
    return df.dropna()

def get_feature_columns():
    """Return the list of feature columns used for modeling"""
    return [
        'RSI', 'MACD', 'MACD_Signal', 'MACD_Diff', 'SMA_20', 'EMA_12', 'EMA_26',
        'ATR', 'Stoch', 'Stoch_Signal', 'BB_Width', 'OBV', 'Price_Change',
        'Price_MA_Ratio'
    ]