# import numpy as np
# import pandas as pd

# # 1. Calculate Volatility (Standard Deviation of Returns)
# def calculate_volatility(stock_data, window=20):
#     stock_data['Returns'] = stock_data['Close'].pct_change()
#     stock_data['Volatility'] = stock_data['Returns'].rolling(window=window).std()
#     return stock_data

# # 2. Check if the stock passes the Volatility filter
# def passes_volatility_filter(stock_data, threshold=0.02):
#     stock_data = calculate_volatility(stock_data)
#     latest_volatility = stock_data['Volatility'].iloc[-1]
#     if latest_volatility > threshold:
#         print(f"Volatility is too high: {latest_volatility:.4f} (Threshold: {threshold})")
#         return False
#     return True

# # 3. Trend Direction Filter: Check if the stock is in an uptrend or downtrend
# def check_trend_direction(stock_data, ma_window=20):
#     stock_data['SMA'] = stock_data['Close'].rolling(window=ma_window).mean()  # Simple Moving Average (SMA)
#     if stock_data['Close'].iloc[-1] < stock_data['SMA'].iloc[-1]:
#         print(f"Stock is in a downtrend (Close: {stock_data['Close'].iloc[-1]} < SMA: {stock_data['SMA'].iloc[-1]})")
#         return False  # Stock is in a downtrend
#     return True  # Stock is in an uptrend

# # 4. Liquidity Filter: Check if the stock has enough trading volume
# def check_liquidity(stock_data, volume_threshold=1000000):
#     average_volume = stock_data['Volume'].mean()
#     if average_volume < volume_threshold:
#         print(f"Liquidity is too low: Average Volume = {average_volume} (Threshold: {volume_threshold})")
#         return False
#     return True

# # 5. Diversification Filter: Ensure no more than 2 stocks from the same sector (optional)
# # Assuming we have sector information available in stock_data['Sector']
# def check_diversification(stocks_in_sector, max_stocks_in_sector=2):
#     if stocks_in_sector > max_stocks_in_sector:
#         print(f"Too many stocks from the same sector: {stocks_in_sector} (Max: {max_stocks_in_sector})")
#         return False
#     return True

# # 6. Risk Filter to apply all conditions
# def passes_risk_filters(stock_data, stocks_in_sector=0, volatility_threshold=0.02, volume_threshold=1000000, ma_window=20):
#     if not passes_volatility_filter(stock_data, volatility_threshold):
#         return False
#     if not check_trend_direction(stock_data, ma_window):
#         return False
#     if not check_liquidity(stock_data, volume_threshold):
#         return False
#     if not check_diversification(stocks_in_sector):
#         return False
#     return True


# risk_filter.py
def calculate_volatility(stock_data):
    stock_data['ATR_Volatility'] = stock_data['ATR'] / stock_data['Close']
    stock_data['BB_Volatility'] = stock_data['BB_Width']
    stock_data['Combined_Volatility'] = (stock_data['ATR_Volatility'] + stock_data['BB_Volatility']) / 2
    return stock_data

# Volatility filter using new combined volatility metric
def passes_volatility_filter(stock_data, threshold=0.03):
    stock_data = calculate_volatility(stock_data)
    latest_vol = stock_data['Combined_Volatility'].iloc[-1]
    if latest_vol > threshold:
        print(f"Volatility too high: {latest_vol:.4f} (Threshold: {threshold})")
        return False
    return True

# Trend filter using MA_Crossover and Price_MA_Ratio
def check_trend_direction(stock_data):
    last_cross = stock_data['MA_Crossover_12_26'].iloc[-1]
    ratio = stock_data['Price_MA_Ratio'].iloc[-1]
    if last_cross == 0 or ratio < 1:
        print(f"Downtrend detected: MA Crossover = {last_cross}, Price/MA Ratio = {ratio:.2f}")
        return False
    return True

# Liquidity filter using 5-day volume average
def check_liquidity(stock_data, volume_threshold=1000000):
    avg_volume = stock_data['Volume'].rolling(window=5).mean().iloc[-1]
    if avg_volume < volume_threshold:
        print(f"Low liquidity: Avg Volume = {avg_volume:.0f} (Threshold = {volume_threshold})")
        return False
    return True

# Diversification filter placeholder (can expand with real sector data)
def check_diversification(stocks_in_sector, max_stocks_in_sector=2):
    if stocks_in_sector > max_stocks_in_sector:
        print(f"Too many stocks from the same sector: {stocks_in_sector} > {max_stocks_in_sector}")
        return False
    return True

# Main combined risk filter
def passes_risk_filters(stock_data, stocks_in_sector=0, volatility_threshold=0.03, volume_threshold=1000000):
    if not passes_volatility_filter(stock_data, volatility_threshold):
        return False
    if not check_trend_direction(stock_data):
        return False
    if not check_liquidity(stock_data, volume_threshold):
        return False
    if not check_diversification(stocks_in_sector):
        return False
    return True
