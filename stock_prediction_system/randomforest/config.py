# Configuration parameters
TICKERS = ['CBA.AX', 'BHP.AX', 'CSL.AX', 'WBC.AX', 'NAB.AX']
RETURN_THRESHOLD = 0.02  # 2% return threshold for buy signal
PREDICTION_PROBABILITY_THRESHOLD = 0.6  # Minimum probability to consider a buy signal
LOOKBACK_YEARS = 10  # Fetch more historical data
DATA_INTERVAL = "1d"  # Weekly data
RANDOM_STATE = 42
INITIAL_CAPITAL = 100000  # Starting capital for backtest

# Risk Management Parameters
RISK_PARAMS = {
    'max_portfolio_risk': 0.02,  # 2% of portfolio risk per trade
    'max_position_size': 0.2,    # 20% of portfolio in single position
    'default_stop_loss': 0.05,   # 5% default stop loss
    'take_profit': 0.1,          # 10% take profit
    'max_sector_exposure': 0.3,  # 30% max sector exposure
    'max_small_cap_exposure': 0.4  # 40% max small cap exposure
}


# Feature interpretation thresholds
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
STOCH_OVERBOUGHT = 80
STOCH_OVERSOLD = 20