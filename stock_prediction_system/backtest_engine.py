import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import ta
import joblib
from datetime import datetime, timedelta
from itertools import cycle
from io import StringIO  # Assuming you need io for something like csv loading
from utils import get_result_path

# Scikit-learn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler

# Imbalanced-learn
from imblearn.over_sampling import SMOTE

# Backtrader
import backtrader as bt
import backtrader.analyzers as btanalyzers

# Custom Config
from config import INITIAL_CAPITAL, PREDICTION_PROBABILITY_THRESHOLD

class MLSignalData(bt.feeds.PandasData):
    """Custom PandasData class to include our ML signals and features"""
    lines = ('buy_signal', 'buy_prob', 'rsi', 'macd', 'macd_signal', 'macd_diff', 
             'sma20', 'ema12', 'ema26', 'atr', 'stoch', 'stoch_signal', 
             'bb_width', 'obv', 'price_change', 'price_ma_ratio')
    
    params = (
        ('datetime', None),
        ('open', 'Open'),
        ('high', 'High'),
        ('low', 'Low'),
        ('close', 'Close'),
        ('volume', 'Volume'),
        ('buy_signal', 'Buy_Prediction'),
        ('buy_prob', 'Buy_Probability'),
        ('rsi', 'RSI'),
        ('macd', 'MACD'),
        ('macd_signal', 'MACD_Signal'),
        ('macd_diff', 'MACD_Diff'),
        ('sma20', 'SMA_20'),
        ('ema12', 'EMA_12'),
        ('ema26', 'EMA_26'),
        ('atr', 'ATR'),
        ('stoch', 'Stoch'),
        ('stoch_signal', 'Stoch_Signal'),
        ('bb_width', 'BB_Width'),
        ('obv', 'OBV'),
        ('price_change', 'Price_Change'),
        ('price_ma_ratio', 'Price_MA_Ratio'),
    )

class MLPredictionStrategy(bt.Strategy):
    """Strategy based on machine learning predictions"""
    params = (
        ('prob_threshold', PREDICTION_PROBABILITY_THRESHOLD),
        ('position_size', 0.2),
        ('stop_loss', 0.05),
        ('take_profit', 0.1),
    )
    
    def __init__(self):
        self.orders = {}
        self.stop_orders = {}
        self.profit_orders = {}
        self.positions_tracker = {}
        self.colors = cycle(['red', 'green', 'blue', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan'])
        self.ticker_colors = {}
        
        for i, d in enumerate(self.datas):
            ticker = d._name
            self.ticker_colors[ticker] = next(self.colors)
            self.positions_tracker[ticker] = {'buy_price': None, 'stop_price': None, 'profit_price': None}
    
    def log(self, txt, dt=None):
        dt = dt or self.datetime.date(0)
        print(f'{dt.isoformat()} {txt}')
    
    def next(self):
        for i, d in enumerate(self.datas):
            ticker = d._name
            pos = self.getposition(d).size
            
            if pos > 0:
                continue
            
            if d.buy_prob[0] >= self.p.prob_threshold and d.buy_signal[0] == 1:
                value = self.broker.getvalue()
                size = int((value * self.p.position_size) / d.close[0])
                
                if size > 0:
                    self.log(f'BUY CREATE - {ticker} at {d.close[0]:.2f}')
                    buy_order = self.buy(data=d, size=size)
                    self.orders[buy_order.ref] = {'ticker': ticker, 'price': d.close[0]}
                    self.positions_tracker[ticker]['buy_price'] = d.close[0]
                    self.positions_tracker[ticker]['stop_price'] = d.close[0] * (1 - self.p.stop_loss)
                    self.positions_tracker[ticker]['profit_price'] = d.close[0] * (1 + self.p.take_profit)
    
    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'BUY EXECUTED - Price: {order.executed.price:.2f}, Size: {order.executed.size}, Cost: {order.executed.value:.2f}')
                
                ticker = self.orders[order.ref]['ticker']
                data = None
                for d in self.datas:
                    if d._name == ticker:
                        data = d
                        break
                
                if data:
                    stop_price = order.executed.price * (1 - self.p.stop_loss)
                    stop_order = self.sell(data=data, size=order.executed.size, exectype=bt.Order.Stop, price=stop_price)
                    self.stop_orders[stop_order.ref] = {'ticker': ticker, 'parent': order.ref}
                    
                    profit_price = order.executed.price * (1 + self.p.take_profit)
                    profit_order = self.sell(data=data, size=order.executed.size, exectype=bt.Order.Limit, price=profit_price)
                    self.profit_orders[profit_order.ref] = {'ticker': ticker, 'parent': order.ref}
            
            elif order.issell():
                self.log(f'SELL EXECUTED - Price: {order.executed.price:.2f}, Size: {order.executed.size}, Cost: {order.executed.value:.2f}')
                
                if order.ref in self.stop_orders:
                    parent_ref = self.stop_orders[order.ref]['parent']
                    for ref, order_info in self.profit_orders.items():
                        if order_info['parent'] == parent_ref:
                            for o in self.broker.orders:
                                if o.ref == ref and not o.executed.size:
                                    self.broker.cancel(o)
                
                elif order.ref in self.profit_orders:
                    parent_ref = self.profit_orders[order.ref]['parent']
                    for ref, order_info in self.stop_orders.items():
                        if order_info['parent'] == parent_ref:
                            for o in self.broker.orders:
                                if o.ref == ref and not o.executed.size:
                                    self.broker.cancel(o)
            
            if order.issell():
                ticker = None
                for ref_dict in [self.orders, self.stop_orders, self.profit_orders]:
                    if order.ref in ref_dict:
                        ticker = ref_dict[order.ref]['ticker']
                        break
                
                if ticker:
                    buy_price = None
                    if order.ref in self.stop_orders:
                        parent_ref = self.stop_orders[order.ref]['parent']
                        buy_price = self.orders[parent_ref]['price']
                    elif order.ref in self.profit_orders:
                        parent_ref = self.profit_orders[order.ref]['parent']
                        buy_price = self.orders[parent_ref]['price']
                    
                    if buy_price:
                        profit_pct = (order.executed.price - buy_price) / buy_price * 100
                        self.log(f'TRADE RESULT - {ticker}: {profit_pct:.2f}%')
        
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log(f'Order Canceled/Margin/Rejected - {order.status}')
    
    def stop(self):
        self.log(f'Final Portfolio Value: {self.broker.getvalue():.2f}')
        roi = (self.broker.getvalue() / INITIAL_CAPITAL - 1) * 100
        self.log(f'Return on Investment: {roi:.2f}%')

def prepare_backtrader_data(data, model_data, backtest_period_days=365):
    """Prepare data for Backtrader backtesting"""
    bt_data = data.copy()
    
    # Get model components
    model = model_data['model']
    features = model_data['features']
    
    # Use the pipeline to transform features (includes scaling)
    bt_data['Buy_Prediction'] = model.predict(bt_data[features])
    bt_data['Buy_Probability'] = model.predict_proba(bt_data[features])[:, 1]
    
    # Convert 'Date' column to datetime if it's not already
    bt_data['Date'] = pd.to_datetime(bt_data['Date'], utc=True)
    
    # Filter to the backtest period
    end_date = bt_data['Date'].max()
    start_date = end_date - pd.Timedelta(days=backtest_period_days)
    bt_data = bt_data[bt_data['Date'] >= start_date].copy()
    
    # Ensure datetime is in the right format and set as index
    bt_data['datetime'] = pd.to_datetime(bt_data['Date'])
    bt_data.set_index('datetime', inplace=True)
    bt_data.drop('Date', axis=1, inplace=True)
    
    return bt_data

def run_backtest(bt_data, initial_capital=INITIAL_CAPITAL):
    """Run the backtest with Backtrader"""
    cerebro = bt.Cerebro()
    cerebro.addstrategy(MLPredictionStrategy)
    cerebro.broker.setcash(initial_capital)
    cerebro.broker.setcommission(commission=0.001)
    
    cerebro.addanalyzer(btanalyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(btanalyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(btanalyzers.Returns, _name='returns')
    cerebro.addanalyzer(btanalyzers.TradeAnalyzer, _name='trades')
    
    for ticker in bt_data['Ticker'].unique():
        ticker_data = bt_data[bt_data['Ticker'] == ticker].copy()
        ticker_data = ticker_data[~ticker_data.index.duplicated(keep='first')]
        data = MLSignalData(dataname=ticker_data, name=ticker)
        cerebro.adddata(data)
    
    results = cerebro.run()
    strategy = results[0]
    
    end_portfolio_value = cerebro.broker.getvalue()
    
    sharpe_ratio = strategy.analyzers.sharpe.get_analysis().get('sharperatio', 0)
    if np.isnan(sharpe_ratio):
        sharpe_ratio = 0
    
    drawdown = strategy.analyzers.drawdown.get_analysis()
    max_drawdown = drawdown.get('max', {}).get('drawdown', 0)
    
    returns = strategy.analyzers.returns.get_analysis()
    roi = returns.get('rtot', 0) * 100
    annual_roi = returns.get('rnorm100', 0)
    
    trade_analysis = strategy.analyzers.trades.get_analysis()
    
    result = {
        'initial_value': initial_capital,
        'final_value': end_portfolio_value,
        'roi': roi,
        'annual_roi': annual_roi,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'total_trades': trade_analysis.get('total', {}).get('total', 0),
        'win_rate': (trade_analysis.get('won', {}).get('total', 0)) / trade_analysis.get('total', {}).get('total', 1) * 100,
        'avg_profit': trade_analysis.get('won', {}).get('pnl', {}).get('average', 0) if trade_analysis.get('won', {}).get('total', 0) > 0 else 0,
        'avg_loss': trade_analysis.get('lost', {}).get('pnl', {}).get('average', 0) if trade_analysis.get('lost', {}).get('total', 0) > 0 else 0,
    }
    
    plt.rcParams['figure.figsize'] = [14, 8]
    plt.rcParams['axes.grid'] = True
    cerebro.plot(style='candlestick', volume=False, iplot=False)
    plot_path = get_result_path('backtest_results.png')
    plt.savefig(plot_path)
    plt.close()
    
    result['plot_file'] = plot_path
    return result