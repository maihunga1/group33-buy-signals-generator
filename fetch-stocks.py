import yfinance as yf
import pandas as pd
import os
from datetime import datetime, timedelta

tickers = ['CBA.AX', 'BHP.AX', 'CSL.AX', 'WBC.AX', 'NAB.AX']

end_date = datetime.today().strftime('%Y-%m-%d')
start_date = (datetime.today() - timedelta(days=30)).strftime('%Y-%m-%d')

def clean_number(value):
    return round(value, 2) if isinstance(value, (int, float)) else value

stock_file = "data/aus_stock_data.csv"
if os.path.exists(stock_file):
    stock_df = pd.read_csv(stock_file)
else:
    stock_df = pd.DataFrame()

for ticker in tickers:
    stock = yf.Ticker(ticker)
    info = stock.info

    new_stock_data = {
        "Ticker": ticker,
        "Company Name": info.get("longName", "N/A"),
        "Current Price": clean_number(info.get("currentPrice", 0)),
        "Previous Close": clean_number(info.get("previousClose", 0)),
        "Market Cap": clean_number(info.get("marketCap", 0)),
        "Return YTD": clean_number(info.get("ytdReturn", 0)),
        "P/E Ratio": clean_number(info.get("trailingPE", 0)),
        "52-Week High": clean_number(info.get("fiftyTwoWeekHigh", 0)),
        "52-Week Low": clean_number(info.get("fiftyTwoWeekLow", 0)),
    }

    if not stock_df.empty and ticker in stock_df["Ticker"].values:
        stock_df.update(pd.DataFrame([new_stock_data]))
    else:
        stock_df = pd.concat([stock_df, pd.DataFrame([new_stock_data])], ignore_index=True)

    hist_file = f"data/{ticker}_historical_data.csv"
    
    if os.path.exists(hist_file):
        existing_hist = pd.read_csv(hist_file, parse_dates=["Date"])
        last_date = existing_hist["Date"].max().strftime('%Y-%m-%d')
    else:
        existing_hist = pd.DataFrame()
        last_date = start_date 

    hist = stock.history(start=last_date, end=end_date)
    hist = hist[['Open', 'High', 'Low', 'Close', 'Volume']].applymap(clean_number)
    hist.insert(0, "Ticker", ticker)

    if not hist.empty:
        hist.fillna(method='ffill', inplace=True)
        hist.index = hist.index.strftime('%Y-%m-%d')
        hist.reset_index(inplace=True)
        hist.rename(columns={"index": "Date"}, inplace=True)

        combined_hist = pd.concat([existing_hist, hist]).drop_duplicates(subset=["Date"]).reset_index(drop=True)
        combined_hist.to_csv(hist_file, index=False)

stock_df.to_csv(stock_file, index=False)

print("✅ Data updated and appended to existing CSV files.")

//fillna with mode