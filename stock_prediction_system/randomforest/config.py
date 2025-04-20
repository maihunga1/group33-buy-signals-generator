# Configuration parameters
TICKERS = ['CBA.AX', 'BHP.AX', 'CSL.AX', 'WBC.AX', 'NAB.AX']
RETURN_THRESHOLD = 0.02  # 2% return threshold for buy signal
PREDICTION_PROBABILITY_THRESHOLD = 0.6  # Minimum probability to consider a buy signal
LOOKBACK_YEARS = 10  # Fetch more historical data
DATA_INTERVAL = "1d"  # Daily data
RANDOM_STATE = 42
INITIAL_CAPITAL = 100000  # Starting capital for backtest

# Feature interpretation thresholds
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
STOCH_OVERBOUGHT = 80
STOCH_OVERSOLD = 20