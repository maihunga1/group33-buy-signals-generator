import json
import yfinance as yf
from datetime import datetime, timedelta
from log_utils import log_message


def stock_data_frame_to_json_by_all_timeframe(ticker):
    stock = yf.Ticker(ticker)
    end_date = datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.today() - timedelta(days=30)).strftime('%Y-%m-%d')

    hist_daily = stock.history(start=start_date, end=end_date, interval="1d")
    hist_weekly = stock.history(start=start_date, end=end_date, interval="1wk")
    hist_hourly = stock.history(period="7d", interval="1h")
    hist_1m = stock.history(period="1d", interval="5m")

    json_data_hist_daily = hist_daily.to_json(orient='records')
    json_data_hist_weekly = hist_weekly.to_json(orient='records')
    json_data_hist_hourly = hist_hourly.to_json(orient='records')
    json_data_hist_1m = hist_1m.to_json(orient='records')

    json_data_hist_daily = json.loads(json_data_hist_daily)
    json_data_hist_weekly = json.loads(json_data_hist_weekly)
    json_data_hist_hourly = json.loads(json_data_hist_hourly)
    json_data_hist_1m = json.loads(json_data_hist_1m)

    for idx, record in enumerate(json_data_hist_daily):
        record['date'] = hist_daily.index[idx]

    for idx, record in enumerate(json_data_hist_weekly):
        record['date'] = hist_weekly.index[idx]

    for idx, record in enumerate(json_data_hist_hourly):
        record['date'] = hist_hourly.index[idx]

    for idx, record in enumerate(json_data_hist_1m):
        record['date'] = hist_1m.index[idx]

    log_message({
        ticker: {
            "daily": json_data_hist_daily,
            "weekly": json_data_hist_weekly,
            "hourly": json_data_hist_hourly,
            "1_minute": json_data_hist_1m
        }})

    return ({
        ticker: {
            "daily": json_data_hist_daily,
            "weekly": json_data_hist_weekly,
            "hourly": json_data_hist_hourly,
            "1_minute": json_data_hist_1m
    }})