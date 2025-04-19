import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from feature_engineering import get_feature_columns
from utils import get_plot_path

def plot_confusion_matrix(y_test, y_pred):
    """Plot confusion matrix"""
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Buy', 'Buy'], 
                yticklabels=['No Buy', 'Buy'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plot_path = get_plot_path('confusion_matrix.png')
    plt.savefig(plot_path)
    plt.close()
    return plot_path

def plot_feature_importance(model):
    """Plot feature importance"""
    plt.figure(figsize=(10, 8))
    features = get_feature_columns()
    feature_importance = pd.DataFrame({
        'Feature': features,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    sns.barplot(x='Importance', y='Feature', data=feature_importance)
    plt.title('Feature Importance')
    plt.tight_layout()
    plot_path = get_plot_path('feature_importance.png')
    plt.savefig(plot_path)
    plt.close()
    return plot_path

def plot_current_predictions(latest_predictions, threshold):
    """Plot current predictions"""
    plt.figure(figsize=(12, 8))
    ax = sns.barplot(x='Ticker', y='Buy_Probability', hue='Ticker', 
                     data=latest_predictions, legend=False)
    
    for i, p in enumerate(ax.patches):
        if latest_predictions['Buy_Probability'].iloc[i] >= threshold:
            p.set_facecolor('green')
        else:
            p.set_facecolor('red')
    
    for i, p in enumerate(ax.patches):
        ax.annotate(f"{latest_predictions['Buy_Probability'].iloc[i]:.2f}", 
                   (p.get_x() + p.get_width() / 2., p.get_height()),
                   ha='center', va='bottom', fontsize=10)
    
    plt.axhline(y=threshold, color='blue', linestyle='--', 
                label=f'Buy Threshold ({threshold})')
    plt.title('Buy Signal Probabilities for Current Week')
    plt.xlabel('Stock')
    plt.ylabel('Probability')
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plot_path = get_plot_path('current_predictions.png')
    plt.savefig(plot_path)
    plt.close()
    return plot_path

def plot_historical_analysis(data, latest_predictions):
    """Plot historical price charts with signals"""
    recent_data = []
    for ticker in data['Ticker'].unique():
        ticker_data = data[data['Ticker'] == ticker].tail(12)
        recent_data.append(ticker_data)
    
    recent_df = pd.concat(recent_data)
    unique_tickers = data['Ticker'].unique()
    
    fig, axes = plt.subplots(len(unique_tickers), 1, 
                            figsize=(14, 4*len(unique_tickers)), sharex=False)
    
    for i, ticker in enumerate(unique_tickers):
        ax = axes[i]
        stock_data = recent_df[recent_df['Ticker'] == ticker].copy()
        
        ax.plot(stock_data['Date'], stock_data['Close'], label='Close Price', color='blue')
        
        buy_signals = stock_data[stock_data['Buy_Signal'] == 1]
        if not buy_signals.empty:
            ax.scatter(buy_signals['Date'], buy_signals['Close'], color='green', s=100, 
                      marker='^', label='Historical Buy Signal')
        
        latest_pred = latest_predictions[latest_predictions['Ticker'] == ticker]
        if not latest_pred.empty:
            last_date = stock_data['Date'].iloc[-1]
            last_close = stock_data['Close'].iloc[-1]
            
            if latest_pred['Buy_Prediction'].values[0] == 1:
                ax.scatter([last_date], [last_close], color='lime', s=200, marker='*', 
                          label='Current Buy Prediction')
                ax.annotate(f"{latest_pred['Buy_Probability'].values[0]:.2f}", 
                            (last_date, last_close),
                            xytext=(10, 10), textcoords='offset points')
        
        if 'SMA_20' in stock_data.columns:
            ax.plot(stock_data['Date'], stock_data['SMA_20'], 
                   label='20-period MA', color='orange', linestyle='--')
        
        ax.set_title(f"{ticker} - Recent Price History")
        ax.set_ylabel('Price')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left')
    
    plt.tight_layout()
    plot_path = get_plot_path('historical_analysis.png')
    plt.savefig(plot_path)
    plt.close()
    return plot_path

def plot_feature_correlation(data):
    """Plot feature correlation matrix"""
    plt.figure(figsize=(14, 12))
    features = get_feature_columns()
    corr_matrix = data[features].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plot_path = get_plot_path('feature_correlation.png')
    plt.savefig(plot_path)
    plt.close()
    return plot_path