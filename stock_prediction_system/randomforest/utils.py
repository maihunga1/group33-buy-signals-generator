import pandas as pd
from datetime import datetime
import os
import shutil

def setup_folders():
    """Create necessary folders if they don't exist"""
    folders = ['data', 'data/raw', 'data/processed', 'plots', 'results', 'models']
    for folder in folders:
        os.makedirs(folder, exist_ok=True)

def get_data_path(filename, subfolder='raw'):
    """Get full path for data files"""
    return os.path.join('data', subfolder, filename)

def get_plot_path(filename):
    """Get full path for plot files"""
    return os.path.join('plots', filename)

def get_result_path(filename):
    """Get full path for result files"""
    return os.path.join('results', filename)

def get_model_path(filename):
    """Get full path for model files"""
    return os.path.join('models', filename)

def generate_report(latest_predictions, backtest_results, visualizations, model_data):
    """Generate a comprehensive report of the analysis"""
    # Create folders if they don't exist
    setup_folders()
    
    # Copy images to results folder for the report
    report_images_dir = get_result_path('report_images')
    os.makedirs(report_images_dir, exist_ok=True)
    
    # Copy all visualization files to report images directory
    for img_file in visualizations.values():
        if os.path.exists(img_file):
            shutil.copy(img_file, report_images_dir)
    
    # Update paths in the report to point to the copied images
    updated_visualizations = {
        name: os.path.join('report_images', os.path.basename(path))
        for name, path in visualizations.items()
    }

    """Generate a comprehensive report of the analysis"""
    html_report = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Stock Prediction Analysis Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1 {{ color: #2c3e50; }}
            h2 {{ color: #3498db; border-bottom: 1px solid #eee; padding-bottom: 5px; }}
            .ticker-card {{ border: 1px solid #ddd; padding: 15px; margin-bottom: 20px; border-radius: 5px; }}
            .buy-signal {{ background-color: #e8f8f5; }}
            .hold-signal {{ background-color: #fef9e7; }}
            .avoid-signal {{ background-color: #fdedec; }}
            .results-table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
            .results-table th, .results-table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            .results-table th {{ background-color: #f2f2f2; }}
            .image-container {{ margin: 20px 0; text-align: center; }}
            .image-container img {{ max-width: 100%; height: auto; border: 1px solid #ddd; }}
            .footer {{ margin-top: 30px; font-size: 0.8em; color: #7f8c8d; text-align: center; }}
        </style>
    </head>
    <body>
        <h1>Stock Prediction Analysis Report</h1>
        <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <h2>Current Week Predictions</h2>
    """
    
    for _, row in latest_predictions.iterrows():
        signal_class = ""
        if row['Recommendation'] in ['Strong Buy', 'Buy']:
            signal_class = "buy-signal"
        elif row['Recommendation'] == 'Hold':
            signal_class = "hold-signal"
        else:
            signal_class = "avoid-signal"
        
        html_report += f"""
        <div class="ticker-card {signal_class}">
            <h3>{row['Ticker']}</h3>
            <p><strong>Recommendation:</strong> {row['Recommendation']}</p>
            <p><strong>Buy Probability:</strong> {row['Buy_Probability']:.2%}</p>
            <p><strong>Current Price:</strong> ${row['Close']:.2f}</p>
            <p><strong>RSI:</strong> {row['RSI']:.2f}</p>
            <p><strong>MACD:</strong> {row['MACD']:.4f}</p>
        </div>
        """
    
    html_report += f"""
    <h2>Backtest Results (Last {backtest_results.get('backtest_days', 365)} Days)</h2>
    <table class="results-table">
        <tr>
            <th>Metric</th>
            <th>Value</th>
        </tr>
        <tr>
            <td>Initial Portfolio Value</td>
            <td>${backtest_results['initial_value']:,.2f}</td>
        </tr>
        <tr>
            <td>Final Portfolio Value</td>
            <td>${backtest_results['final_value']:,.2f}</td>
        </tr>
        <tr>
            <td>Return on Investment</td>
            <td>{backtest_results['roi']:.2f}%</td>
        </tr>
        <tr>
            <td>Annualized Return</td>
            <td>{backtest_results['annual_roi']:.2f}%</td>
        </tr>
        <tr>
            <td>Sharpe Ratio</td>
            <td>{backtest_results['sharpe_ratio']:.3f}</td>
        </tr>
        <tr>
            <td>Maximum Drawdown</td>
            <td>{backtest_results['max_drawdown']:.2f}%</td>
        </tr>
        <tr>
            <td>Total Trades</td>
            <td>{backtest_results['total_trades']}</td>
        </tr>
        <tr>
            <td>Win Rate</td>
            <td>{backtest_results['win_rate']:.2f}%</td>
        </tr>
        <tr>
            <td>Average Profit</td>
            <td>${backtest_results['avg_profit']:.2f}</td>
        </tr>
        <tr>
            <td>Average Loss</td>
            <td>${backtest_results['avg_loss']:.2f}</td>
        </tr>
    </table>
    """
    
    for title, img_file in visualizations.items():
        html_report += f"""
        <div class="image-container">
            <h3>{title.replace('_', ' ').title()}</h3>
            <img src="{img_file}" alt="{title}">
        </div>
        """
    
    if 'plot_file' in backtest_results:
        html_report += f"""
        <div class="image-container">
            <h3>Backtest Results</h3>
            <img src="{backtest_results['plot_file']}" alt="Backtest Results">
        </div>
        """
    
    html_report += f"""
    <div class="footer">
        <p>Analysis generated by Stock Prediction System using Random Forest Classifier</p>
        <p>Model trained on {model_data['training_date']} with return threshold of 2.00%</p>
    </div>
    </body>
    </html>
    """
    
    report_path = get_result_path('stock_prediction_report.html')
    with open(report_path, 'w') as f:
        f.write(html_report)
    
    print(f"✅ Report saved as '{report_path}'")
    return report_path