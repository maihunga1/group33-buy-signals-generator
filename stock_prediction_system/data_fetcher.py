import yfinance as yf
import pandas as pd
from utils import get_data_path, setup_folders
from config import TICKERS, LOOKBACK_YEARS, DATA_INTERVAL

def fetch_stock_data(tickers, period=f"{LOOKBACK_YEARS}y", interval=DATA_INTERVAL):
    """Fetch stock data from Yahoo Finance"""
    setup_folders()  # Ensure folders exist
    
    print(f"🔄 Fetching {period} of {interval} data for {len(tickers)} stocks...")
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
            df['Ticker'] = ticker
            all_data.append(df)
            
            # Save raw data for each ticker
            raw_data_path = get_data_path(f"{ticker}_raw.csv")
            df.to_csv(raw_data_path, index=False)
            print(f"💾 Saved raw data to {raw_data_path}")
            
        except Exception as e:
            print(f"❌ Error processing {ticker}: {e}")
    
    # Save combined raw data
    combined_data = pd.concat(all_data) if all_data else pd.DataFrame()
    if not combined_data.empty:
        combined_raw_path = get_data_path("all_tickers_raw.csv")
        combined_data.to_csv(combined_raw_path, index=False)
        print(f"💾 Saved combined raw data to {combined_raw_path}")
    
    return combined_data