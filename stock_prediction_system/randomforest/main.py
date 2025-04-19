import pandas as pd
from data_fetcher import fetch_stock_data
from feature_engineering import create_features
from model_training import train_model
from signal_generator import generate_predictions
from visualization import (plot_confusion_matrix, plot_feature_importance,
                         plot_current_predictions, plot_historical_analysis,
                         plot_feature_correlation)
from backtest_engine import prepare_backtrader_data, run_backtest
from utils import generate_report
from config import TICKERS, PREDICTION_PROBABILITY_THRESHOLD
import os

def main():
    try:
        from utils import setup_folders, get_data_path
        setup_folders()
        
        # Step 1: Fetch and prepare data
        processed_data_path = get_data_path("processed_data.csv", subfolder='processed')
        if os.path.exists(processed_data_path):
            print("🔍 Loading existing processed data...")
            weekly_data = pd.read_csv(processed_data_path)
            print(f"✅ Loaded processed data from {processed_data_path}")
        else:
            # Fetch and process data if not exists
            raw_data = fetch_stock_data(TICKERS)
            weekly_data = create_features(raw_data)
        
        # Step 2: Train model
        model_data = train_model(weekly_data)
        
        # Step 3: Generate current predictions
        latest_predictions = generate_predictions(model_data, weekly_data)
        print("\n📈 Current Week Predictions:")
        print(latest_predictions[['Ticker', 'Close', 'Buy_Probability', 'Recommendation']])
        
        # Step 4: Create visualizations
        visualizations = {
            'current_predictions': plot_current_predictions(latest_predictions, PREDICTION_PROBABILITY_THRESHOLD),
            'historical_analysis': plot_historical_analysis(weekly_data, latest_predictions),
            'feature_correlation': plot_feature_correlation(weekly_data)
        }
        
        # Step 5: Run backtest
        bt_data = prepare_backtrader_data(weekly_data, model_data)
        backtest_results = run_backtest(bt_data)
        backtest_results['backtest_days'] = 365
        
        # Step 6: Generate report
        report_file = generate_report(latest_predictions, backtest_results, visualizations, model_data)
        
        print(f"\n🎉 Analysis complete! Report generated: {report_file}")
        
    except Exception as e:
        print(f"❌ Error in main execution: {e}")
        raise

if __name__ == "__main__":
    main()