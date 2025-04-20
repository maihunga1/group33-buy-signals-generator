import yfinance as yf
import pandas as pd
import ta

# Configuration parameters
TICKERS = ['CBA.AX', 'BHP.AX', 'CSL.AX', 'WBC.AX', 'NAB.AX']
LOOKBACK_YEARS = 10
DATA_INTERVAL = "1wk"
OUTPUT_CSV = 'combined_stock_data.csv'

def fetch_and_calculate(tickers, period=f"{LOOKBACK_YEARS}y", interval=DATA_INTERVAL):
    """Fetch stock data and calculate technical indicators, then save to CSV."""
    all_data = []
    
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period=period, interval=interval)
            if df.empty:
                print(f"⚠️ No data found for {ticker}, skipping.")
                continue
                
            df.reset_index(inplace=True)
            print(f"✅ Fetched {len(df)} periods for {ticker}")
            
            # Add technical indicators
            df['SMA_20'] = ta.trend.SMAIndicator(df['Close'], window=20).sma_indicator()
            df['EMA_12'] = ta.trend.EMAIndicator(df['Close'], window=12).ema_indicator()
            df['EMA_26'] = ta.trend.EMAIndicator(df['Close'], window=26).ema_indicator()
            df['MACD'] = ta.trend.MACD(df['Close']).macd()
            df['MACD_Signal'] = ta.trend.MACD(df['Close']).macd_signal()
            df['MACD_Diff'] = ta.trend.MACD(df['Close']).macd_diff()
            df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
            df['Stoch'] = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close']).stoch()
            df['Stoch_Signal'] = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close']).stoch_signal()
            df['ATR'] = ta.volatility.AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close']).average_true_range()
            df['BB_High'] = ta.volatility.BollingerBands(df['Close']).bollinger_hband()
            df['BB_Low'] = ta.volatility.BollingerBands(df['Close']).bollinger_lband()
            df['BB_Width'] = (df['BB_High'] - df['BB_Low']) / df['Close']
            df['OBV'] = ta.volume.OnBalanceVolumeIndicator(df['Close'], df['Volume']).on_balance_volume()
            df['Price_Change'] = df['Close'].pct_change()
            df['Price_MA_Ratio'] = df['Close'] / df['SMA_20']
            
            # 1. Volume change over 5 days
            df['Volume_5D_Avg'] = df['Volume'].rolling(window=5).mean()
            df['Volume_Change'] = df['Volume'] / df['Volume_5D_Avg']

            # 2. Return over 5 and 20 days
            df['Return_5D'] = df['Close'].pct_change(periods=5)
            df['Return_20D'] = df['Close'].pct_change(periods=20)

            # 3. MA Crossover signal (1 if EMA_12 > EMA_26 else 0)
            df['MA_Crossover_12_26'] = (df['EMA_12'] > df['EMA_26']).astype(int)

            
            # Add ticker column
            df['Ticker'] = ticker
            all_data.append(df)
            
        except Exception as e:
            print(f"❌ Error processing {ticker}: {e}")
    
    # Combine all stock data
    combined_data = pd.concat(all_data, ignore_index=True)
    
    # Drop missing values
    combined_data.dropna(inplace=True)
    
    # Save to CSV
    combined_data.to_csv(OUTPUT_CSV, index=False)
    print(f"📊 Combined data saved to {OUTPUT_CSV}")

    return combined_data  # Return the combined DataFrame
