import yfinance as yf
import pandas as pd
import numpy as np
import ta  # for technical indicators
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from datetime import datetime

# Configuration parameters
TICKERS = ['CBA.AX', 'BHP.AX', 'CSL.AX', 'WBC.AX', 'NAB.AX']
RETURN_THRESHOLD = 0.02  # 2% return threshold for buy signal
PREDICTION_PROBABILITY_THRESHOLD = 0.6  # Minimum probability to consider a buy signal
LOOKBACK_YEARS = 3  # Fetch more historical data
DATA_INTERVAL = "1wk"  # Weekly data
RANDOM_STATE = 42

def fetch_and_prepare_data(tickers, period=f"{LOOKBACK_YEARS}y", interval=DATA_INTERVAL):
    """Fetch stock data and prepare features"""
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
            
            # Add technical indicators (expanded set)
            # Trend indicators
            df['SMA_20'] = ta.trend.SMAIndicator(df['Close'], window=20).sma_indicator()
            df['EMA_12'] = ta.trend.EMAIndicator(df['Close'], window=12).ema_indicator()
            df['EMA_26'] = ta.trend.EMAIndicator(df['Close'], window=26).ema_indicator()
            df['MACD'] = ta.trend.MACD(df['Close']).macd()
            df['MACD_Signal'] = ta.trend.MACD(df['Close']).macd_signal()
            df['MACD_Diff'] = ta.trend.MACD(df['Close']).macd_diff()
            
            # Momentum indicators
            df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
            df['Stoch'] = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close']).stoch()
            df['Stoch_Signal'] = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close']).stoch_signal()
            
            # Volatility indicators
            df['ATR'] = ta.volatility.AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close']).average_true_range()
            df['BB_High'] = ta.volatility.BollingerBands(df['Close']).bollinger_hband()
            df['BB_Low'] = ta.volatility.BollingerBands(df['Close']).bollinger_lband()
            df['BB_Width'] = (df['BB_High'] - df['BB_Low']) / df['Close']
            
            # Volume indicators
            df['OBV'] = ta.volume.OnBalanceVolumeIndicator(df['Close'], df['Volume']).on_balance_volume()
            
            # Price-based features
            df['Price_Change'] = df['Close'].pct_change()
            df['Price_MA_Ratio'] = df['Close'] / df['SMA_20']
            
            # Target creation - forward returns
            df['Return_1w'] = df['Close'].shift(-1) / df['Close'] - 1
            df['Buy_Signal'] = (df['Return_1w'] > RETURN_THRESHOLD).astype(int)
            
            # Meta information
            df['Ticker'] = ticker
            all_data.append(df)
            
        except Exception as e:
            print(f"❌ Error processing {ticker}: {e}")
    
    # Combine all stock data
    combined_data = pd.concat(all_data)
    
    # Drop missing values
    combined_data.dropna(inplace=True)
    print(f"📊 Combined dataset shape: {combined_data.shape}")
    
    return combined_data

def train_improved_model(weekly_data):
    """Train an improved model with hyperparameter tuning and class imbalance handling"""
    # Define our feature set based on technical analysis
    features = [
        'RSI', 'MACD', 'MACD_Signal', 'MACD_Diff', 'SMA_20', 'EMA_12', 'EMA_26',
        'ATR', 'Stoch', 'Stoch_Signal', 'BB_Width', 'OBV', 'Price_Change',
        'Price_MA_Ratio'
    ]
    
    # Create X and y
    X = weekly_data[features]
    y = weekly_data['Buy_Signal']
    
    # Check class distribution
    class_counts = y.value_counts()
    print("\n🔢 Class distribution:")
    print(f"Buy signals (1): {class_counts.get(1, 0)} ({class_counts.get(1, 0)/len(y)*100:.1f}%)")
    print(f"No-buy signals (0): {class_counts.get(0, 0)} ({class_counts.get(0, 0)/len(y)*100:.1f}%)")
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data - using stratified sampling to maintain class proportions
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    
    # Handle class imbalance with SMOTE (only if we have enough samples of minority class)
    if class_counts.get(1, 0) >= 5:  # Need some minimum samples for SMOTE
        print("🔄 Applying SMOTE to balance classes...")
        smote = SMOTE(random_state=RANDOM_STATE)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        
        # Check new class distribution
        print(f"Training data after SMOTE: {len(y_train_resampled)} samples")
        print(f"Class distribution after SMOTE: {np.bincount(y_train_resampled)}")
    else:
        print("⚠️ Not enough minority samples for SMOTE, using original data")
        X_train_resampled, y_train_resampled = X_train, y_train
    
    # Hyperparameter tuning
    print("\n🔍 Performing hyperparameter tuning...")
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'class_weight': [None, 'balanced']
    }
    
    # Use a smaller grid for demonstration (in practice, you'd use the full grid above)
    simple_param_grid = {
        'n_estimators': [100],
        'max_depth': [None],
        'class_weight': [None, 'balanced']
    }
    
    rf = RandomForestClassifier(random_state=RANDOM_STATE)
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=simple_param_grid,  # Use simple_param_grid for faster execution
        cv=3,
        scoring='f1',  # Focus on F1 score to balance precision and recall
        n_jobs=-1
    )
    
    grid_search.fit(X_train_resampled, y_train_resampled)
    
    print(f"Best parameters: {grid_search.best_params_}")
    best_model = grid_search.best_estimator_
    
    # Evaluate the model
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]  # Probabilities for class 1
    
    print("\n📊 Model Evaluation:")
    print(classification_report(y_test, y_pred))
    
    # Calculate ROC AUC
    try:
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        print(f"ROC AUC Score: {roc_auc:.3f}")
    except:
        print("Could not calculate ROC AUC Score")
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Buy', 'Buy'], 
                yticklabels=['No Buy', 'Buy'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    # Feature importance
    plt.figure(figsize=(10, 8))
    feature_importance = pd.DataFrame({
        'Feature': features,
        'Importance': best_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    sns.barplot(x='Importance', y='Feature', data=feature_importance)
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()
    
    # Save model and scaler
    model_data = {
        'model': best_model,
        'scaler': scaler,
        'features': features,
        'threshold': RETURN_THRESHOLD,
        'training_date': datetime.now().strftime('%Y-%m-%d'),
        'feature_importance': feature_importance.to_dict(),
    }
    
    joblib.dump(model_data, 'stock_prediction_model.pkl')
    print("✅ Model saved as 'stock_prediction_model.pkl'")
    
    return model_data

def generate_predictions(model_data, weekly_data):
    """Generate predictions for latest data"""
    latest = weekly_data.groupby("Ticker").tail(1).copy()
    
    # Get model components
    model = model_data['model']
    scaler = model_data['scaler']
    features = model_data['features']
    
    # Prepare features
    latest_scaled = scaler.transform(latest[features])
    
    # Get both class prediction and probability
    latest['Buy_Prediction'] = model.predict(latest_scaled)
    latest['Buy_Probability'] = model.predict_proba(latest_scaled)[:, 1]
    latest['Recommendation'] = latest['Buy_Probability'].apply(
        lambda x: 'Strong Buy' if x > 0.75 else
                  'Buy' if x > PREDICTION_PROBABILITY_THRESHOLD else
                  'Hold' if x > 0.4 else
                  'Avoid'
    )
    
    return latest

def visualize_predictions(latest_predictions, weekly_data, model_data):
    """Create visualizations for predictions"""
    # 1. Create a detailed current prediction visualization
    plt.figure(figsize=(12, 8))
    
    # Fix for deprecated seaborn warning
    ax = sns.barplot(x='Ticker', y='Buy_Probability', hue='Ticker', data=latest_predictions, legend=False)
    
    # Color bars based on threshold
    for i, p in enumerate(ax.patches):
        if latest_predictions['Buy_Probability'].iloc[i] >= PREDICTION_PROBABILITY_THRESHOLD:
            p.set_facecolor('green')
        else:
            p.set_facecolor('red')
    
    # Add probability values on top of bars
    for i, p in enumerate(ax.patches):
        ax.annotate(f"{latest_predictions['Buy_Probability'].iloc[i]:.2f}", 
                   (p.get_x() + p.get_width() / 2., p.get_height()),
                   ha='center', va='bottom', fontsize=10)
    
    # Add threshold line
    plt.axhline(y=PREDICTION_PROBABILITY_THRESHOLD, color='blue', linestyle='--', 
                label=f'Buy Threshold ({PREDICTION_PROBABILITY_THRESHOLD})')
    
    plt.title('Buy Signal Probabilities for Current Week')
    plt.xlabel('Stock')
    plt.ylabel('Probability')
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig('current_predictions.png')
    plt.close()
    
    # 2. Create historical price chart with buy signals
    # Get last 12 weeks of data for each stock
    recent_data = []
    
    for ticker in weekly_data['Ticker'].unique():
        ticker_data = weekly_data[weekly_data['Ticker'] == ticker].tail(12)
        recent_data.append(ticker_data)
    
    recent_df = pd.concat(recent_data)
    
    # Create subplots - one for each stock
    unique_tickers = weekly_data['Ticker'].unique()
    fig, axes = plt.subplots(len(unique_tickers), 1, figsize=(14, 4*len(unique_tickers)), sharex=False)
    
    for i, ticker in enumerate(unique_tickers):
        ax = axes[i]
        stock_data = recent_df[recent_df['Ticker'] == ticker].copy()
        
        # Plot close price
        ax.plot(stock_data['Date'], stock_data['Close'], label='Close Price', color='blue')
        
        # Highlight buy signals from historical data
        buy_signals = stock_data[stock_data['Buy_Signal'] == 1]
        if not buy_signals.empty:
            ax.scatter(buy_signals['Date'], buy_signals['Close'], color='green', s=100, 
                      marker='^', label='Historical Buy Signal')
        
        # Plot latest prediction
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
        
        # Add moving average
        if 'SMA_20' in stock_data.columns:
            ax.plot(stock_data['Date'], stock_data['SMA_20'], 
                   label='20-period MA', color='orange', linestyle='--')
        
        ax.set_title(f"{ticker} - Recent Price History")
        ax.set_ylabel('Price')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left')
    
    plt.tight_layout()
    plt.savefig('historical_analysis.png')
    plt.close()
    
    # 3. Correlation matrix of features
    plt.figure(figsize=(14, 12))
    feature_cols = model_data['features']
    corr_matrix = weekly_data[feature_cols].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig('feature_correlation.png')
    plt.close()
    
    return {
        'current_predictions': 'current_predictions.png',
        'historical_analysis': 'historical_analysis.png',
        'feature_correlation': 'feature_correlation.png'
    }

def backtest_strategy(weekly_data, model_data, lookback_periods=52):
    """Backtest the prediction strategy over the past year"""
    print("\n📈 Running backtest simulation...")
    
    # Define starting and ending points
    end_index = weekly_data.groupby('Ticker').tail(1).index.max()
    
    # Create copy of data for backtest
    backtest_data = weekly_data.copy()
    
    # We'll track:
    # 1. Performance of buying all stocks (equal weight portfolio)
    # 2. Performance of model-recommended stocks
    # 3. Performance of stocks avoided by the model
    
    results = []
    
    # For each week in our backtest period
    unique_dates = sorted(backtest_data['Date'].unique())
    
    if len(unique_dates) <= lookback_periods:
        lookback_periods = len(unique_dates) - 10  # Use most of the data but leave some for training
    
    backtest_dates = unique_dates[-lookback_periods:]
    
    # Initialize tracking variables
    model_performance = 1.0  # Starting with $1
    market_performance = 1.0  # Equal-weight all stocks
    avoid_performance = 1.0   # Stocks the model says to avoid
    
    for test_date in backtest_dates:
        # Get data up to this date for training
        train_data = backtest_data[backtest_data['Date'] < test_date]
        
        # Skip if we don't have enough training data
        if len(train_data) < 100:
            continue
            
        # Get the test week data
        test_week = backtest_data[backtest_data['Date'] == test_date]
        
        # Train a model on historical data
        X_train = train_data[model_data['features']]
        y_train = train_data['Buy_Signal']
        
        # Scale features
        X_train_scaled = model_data['scaler'].transform(X_train)
        
        # Train the model (use the same parameters as our best model)
        backtest_model = RandomForestClassifier(**model_data['model'].get_params())
        backtest_model.fit(X_train_scaled, y_train)
        
        # Predict on test week
        X_test = test_week[model_data['features']]
        X_test_scaled = model_data['scaler'].transform(X_test)
        
        test_week['Predicted_Probability'] = backtest_model.predict_proba(X_test_scaled)[:, 1]
        test_week['Predicted_Signal'] = (test_week['Predicted_Probability'] > PREDICTION_PROBABILITY_THRESHOLD).astype(int)
        
        # Calculate forward returns (already in the data as Return_1w)
        
        # Initialize values for this iteration
        model_return = 1.0
        avoid_return = 1.0
        
        # Update performance tracking
        # Model picks
        model_picks = test_week[test_week['Predicted_Signal'] == 1]
        if len(model_picks) > 0:
            model_return = model_picks['Return_1w'].mean() + 1  # Average return of picks
            model_performance *= model_return
        
        # Market (all stocks equal weight)
        market_return = test_week['Return_1w'].mean() + 1
        market_performance *= market_return
        
        # Avoided stocks
        avoid_picks = test_week[test_week['Predicted_Signal'] == 0]
        if len(avoid_picks) > 0:
            avoid_return = avoid_picks['Return_1w'].mean() + 1
            avoid_performance *= avoid_return
        
        # Store weekly result
        results.append({
            'Date': test_date,
            'Model_Performance': model_performance,
            'Market_Performance': market_performance,
            'Avoid_Performance': avoid_performance,
            'Weekly_Model_Return': model_return,
            'Weekly_Market_Return': market_return,
            'Weekly_Avoid_Return': avoid_return,
            'Model_Picks': len(model_picks),
            'Total_Stocks': len(test_week)
        })
    
    # Convert results to DataFrame
    backtest_results = pd.DataFrame(results)
    
    # Calculate some key metrics
    if len(backtest_results) > 0:
        total_weeks = len(backtest_results)
        model_win_weeks = sum(backtest_results['Weekly_Model_Return'] > backtest_results['Weekly_Market_Return'])
        
        print(f"\n🔍 Backtest Results ({total_weeks} weeks):")
        print(f"Model Final Performance: {backtest_results['Model_Performance'].iloc[-1]:.2f}x")
        print(f"Market Final Performance: {backtest_results['Market_Performance'].iloc[-1]:.2f}x")
        print(f"Avoid Stocks Performance: {backtest_results['Avoid_Performance'].iloc[-1]:.2f}x")
        print(f"Model Win Rate: {model_win_weeks/total_weeks*100:.1f}% of weeks")
        
        # Annualized return calculation
        weeks_per_year = 52
        years = total_weeks / weeks_per_year
        
        model_annual_return = (backtest_results['Model_Performance'].iloc[-1] ** (1/years)) - 1
        market_annual_return = (backtest_results['Market_Performance'].iloc[-1] ** (1/years)) - 1
        
        print(f"Model Annualized Return: {model_annual_return*100:.1f}%")
        print(f"Market Annualized Return: {market_annual_return*100:.1f}%")
        
        # Plotting backtest results
        plt.figure(figsize=(12, 8))
        plt.plot(backtest_results['Date'], backtest_results['Model_Performance'], label='Model Strategy', linewidth=2)
        plt.plot(backtest_results['Date'], backtest_results['Market_Performance'], label='Market (Equal Weight)', linewidth=2)
        plt.plot(backtest_results['Date'], backtest_results['Avoid_Performance'], label='Avoided Stocks', linewidth=2, linestyle='--')
        
        plt.title('Backtest Performance Comparison')
        plt.ylabel('Cumulative Return (Starting Value = 1)')
        plt.xlabel('Date')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig('backtest_performance.png')
        plt.close()
    else:
        print("⚠️ Insufficient data for backtesting")
    
    return backtest_results if len(backtest_results) > 0 else None

def main():
    """Main function to run the entire process"""
    # 1. Fetch and prepare data
    weekly_data = fetch_and_prepare_data(TICKERS)
    
    # 2. Train the model
    model_data = train_improved_model(weekly_data)
    
    # 3. Generate predictions for the current week
    latest_predictions = generate_predictions(model_data, weekly_data)
    
    # 4. Visualize the predictions
    visualizations = visualize_predictions(latest_predictions, weekly_data, model_data)
    
    # 5. Run backtest
    backtest_results = backtest_strategy(weekly_data, model_data)
    
    # 6. Print final predictions and recommendations
    print("\n🔮 Current Week Buy Predictions:")
    print(latest_predictions[['Ticker', 'Close', 'Buy_Prediction', 'Buy_Probability', 'Recommendation']].sort_values('Buy_Probability', ascending=False))
    
    # Provide a summary
    print("\n📝 Summary:")
    strong_buys = latest_predictions[latest_predictions['Recommendation'] == 'Strong Buy']
    buys = latest_predictions[latest_predictions['Recommendation'] == 'Buy']
    holds = latest_predictions[latest_predictions['Recommendation'] == 'Hold']
    avoids = latest_predictions[latest_predictions['Recommendation'] == 'Avoid']
    
    if len(strong_buys) > 0:
        print(f"Strong Buy: {', '.join(strong_buys['Ticker'])}")
    if len(buys) > 0:
        print(f"Buy: {', '.join(buys['Ticker'])}")
    if len(holds) > 0:
        print(f"Hold: {', '.join(holds['Ticker'])}")
    if len(avoids) > 0:
        print(f"Avoid: {', '.join(avoids['Ticker'])}")
    
    return {
        'model_data': model_data,
        'latest_predictions': latest_predictions,
        'visualizations': visualizations,
        'backtest_results': backtest_results
    }

if __name__ == "__main__":
    main()