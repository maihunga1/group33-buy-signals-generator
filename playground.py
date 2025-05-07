"""NLP for Sentiment Analysis in Financial News with PyTorch and Asyncio"""

# Import required libraries
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from transformers import pipeline
import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np
import requests
from bs4 import BeautifulSoup
from sklearn.metrics import mean_squared_error, r2_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random
import time
import asyncio
import aiohttp

# Download required NLTK resources (with error handling)
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('punkt', quiet=True)
except Exception as e:
    print(f"Error downloading NLTK resources: {e}")
    print("Using fallback approach for NLP processing")

# Text preprocessing function
def preprocess_text(text):
    try:
        tokens = word_tokenize(text.lower())
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
        return ' '.join(tokens)
    except Exception as e:
        print(f"Error in text preprocessing: {e}")
        # Simple fallback text cleaning
        return ' '.join([word.lower() for word in text.split() if len(word) > 2])

# Initialize sentiment analysis model with error handling
try:
    nlp = pipeline("sentiment-analysis", model="yiyanghkust/finbert-tone")
except Exception as e:
    print(f"Error loading FinBERT model: {e}")
    # Fallback to a simpler sentiment analysis approach
    from textblob import TextBlob
    
    def simplified_sentiment(text):
        analysis = TextBlob(text)
        # Convert polarity to positive/negative/neutral classification
        if analysis.sentiment.polarity > 0.1:
            return {'label': 'positive', 'score': analysis.sentiment.polarity}
        elif analysis.sentiment.polarity < -0.1:
            return {'label': 'negative', 'score': analysis.sentiment.polarity}
        else:
            return {'label': 'neutral', 'score': analysis.sentiment.polarity}
    
    nlp = lambda text: [simplified_sentiment(text)]

# Function to analyze sentiment of news articles
def get_sentiment(text):
    try:
        cleaned_text = preprocess_text(text)
        sentiment_result = nlp(cleaned_text)
        # Convert sentiment to a numerical value: 1 for positive, -1 for negative, 0 for neutral
        if sentiment_result[0]['label'] == 'positive':
            return 1
        elif sentiment_result[0]['label'] == 'negative':
            return -1
        else:
            return 0
    except Exception as e:
        print(f"Error in sentiment analysis: {e}")
        # Return neutral sentiment as fallback
        return 0

# Asynchronous function to scrape Yahoo Finance news headlines
async def scrape_yahoo_finance():
    url = 'https://finance.yahoo.com/topic/stock-market-news/'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    # Default headlines in case scraping fails
    default_headlines = [
        "Market shows mixed signals today", 
        "Tech stocks rally on positive earnings", 
        "Investors cautious amid economic uncertainty",
        "Banking sector faces regulatory challenges",
        "Commodities prices surge on supply concerns",
        "Inflation concerns weigh on market sentiment",
        "Central bank maintains interest rates",
        "Global trade tensions affect market outlook",
        "Retail sector reports strong quarterly earnings",
        "Energy stocks decline amid oversupply concerns"
    ]
    
    try:
        # Use aiohttp for async HTTP requests
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, timeout=10) as response:
                if response.status != 200:
                    print(f"Error: Received status code {response.status}")
                    return default_headlines
                
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                
                # Extract the news headlines
                headlines = []
                for item in soup.find_all('h3'):
                    headline = item.get_text(strip=True)
                    if headline:
                        headlines.append(headline)
                
                if headlines:
                    return headlines[:20]  # Return first 20 headlines
                else:
                    print("No headlines found, using default headlines")
                    return default_headlines
    
    except Exception as e:
        print(f"Error scraping Yahoo Finance: {e}")
        return default_headlines

# Define top 5 ASX stocks
asx_stocks = {
    'CBA.AX': 'Commonwealth Bank',
    'BHP.AX': 'BHP Group',
    'CSL.AX': 'CSL Limited',
    'NAB.AX': 'National Australia Bank',
    'WES.AX': 'Wesfarmers'
}

# PyTorch Dataset class for stock data
class StockDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# LSTM Model in PyTorch
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout1 = nn.Dropout(0.2)
        self.fc1 = nn.Linear(hidden_size, 25)
        self.dropout2 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(25, output_size)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout1(out[:, -1, :])  # Take the last time step
        out = torch.relu(self.fc1(out))
        out = self.dropout2(out)
        out = self.fc2(out)
        return out

async def fetch_stock_data(ticker, start_date, end_date, max_retries=3, initial_delay=2):
    """Asynchronously fetch stock data with retry mechanism"""
    import asyncio
    
    retry_delay = initial_delay
    
    for retry in range(max_retries):
        try:
            print(f"Downloading data for {ticker} (attempt {retry+1}/{max_retries})...")
            # Note: yfinance doesn't support async directly, but we can manage the waiting
            stock_data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            
            if stock_data.empty:
                print(f"No data returned for {ticker}")
                if retry < max_retries - 1:
                    print(f"Retrying in {retry_delay} seconds...")
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                    continue
                else:
                    # If all retries fail, return None to generate synthetic data
                    return None
            return stock_data
            
        except Exception as e:
            print(f"Error downloading data for {ticker}: {e}")
            if retry < max_retries - 1:
                print(f"Retrying in {retry_delay} seconds...")
                await asyncio.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                return None
    
    return None

async def prepare_stock_data(ticker):
    # Step 1: Get stock data with async retry mechanism
    stock_data = await fetch_stock_data(ticker, "2022-01-01", "2023-12-31")
    
    if stock_data is None or stock_data.empty:
        print(f"Using synthetic data for {ticker}")
        return generate_synthetic_data()
    
    # Step 2: Get news sentiment (using scraped headlines)
    headlines = await scrape_yahoo_finance()
    cleaned_headlines = [preprocess_text(headline) for headline in headlines]
    sentiments = [get_sentiment(headline) for headline in cleaned_headlines]
    
    # Verify data before proceeding
    if stock_data.empty:
        print(f"Empty data frame for {ticker}. Using synthetic data.")
        return generate_synthetic_data()
    
    # Align sentiment data with stock data (simple approach - random assignment for demo)
    stock_data['Sentiment'] = [random.choice(sentiments) for _ in range(len(stock_data))]
    
    # Create features
    stock_data['Lag_1'] = stock_data['Close'].shift(1)
    stock_data.dropna(inplace=True)
    
    # Check for empty DataFrame after dropna
    if stock_data.empty:
        print(f"Empty data frame after preprocessing for {ticker}. Using synthetic data.")
        return generate_synthetic_data()
    
    # Scale data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(stock_data[['Close', 'Lag_1', 'Open', 'High', 'Low', 'Volume', 'Sentiment']])
    
    # Create sequences
    look_back = min(60, len(scaled_data) - 1)  # Ensure look_back doesn't exceed data length
    X, y = [], []
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i-look_back:i])
        y.append(scaled_data[i, 0])  # Close price is target
    
    X = np.array(X)
    y = np.array(y)
    
    # Verify dimensions
    if len(X) == 0 or len(y) == 0:
        print(f"Empty sequence data for {ticker}. Using synthetic data.")
        return generate_synthetic_data()
    
    return X, y, scaler

def generate_synthetic_data():
    """Generate synthetic data for demonstration when real data is unavailable"""
    print("Generating synthetic stock data for demonstration...")
    
    # Generate 300 days of synthetic price data
    days = 300
    base_price = 100.0
    
    # Create synthetic price movements
    np.random.seed(42)  # For reproducibility
    daily_returns = np.random.normal(0.0005, 0.015, days)
    prices = [base_price]
    
    for ret in daily_returns:
        prices.append(prices[-1] * (1 + ret))
    
    # Create volume and other features
    volumes = np.random.normal(1000000, 300000, days)
    sentiments = np.random.choice([-1, 0, 1], days)
    
    # Create DataFrame with synthetic data
    dates = pd.date_range(start='2022-01-01', periods=days)
    df = pd.DataFrame({
        'Close': prices[1:],
        'Open': [price * (1 + np.random.normal(-0.002, 0.003)) for price in prices[1:]],
        'High': [price * (1 + abs(np.random.normal(0, 0.006))) for price in prices[1:]],
        'Low': [price * (1 - abs(np.random.normal(0, 0.006))) for price in prices[1:]],
        'Volume': volumes,
        'Sentiment': sentiments
    }, index=dates)
    
    # Calculate lag feature
    df['Lag_1'] = df['Close'].shift(1)
    df.dropna(inplace=True)
    
    # Scale data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[['Close', 'Lag_1', 'Open', 'High', 'Low', 'Volume', 'Sentiment']])
    
    # Create sequences
    look_back = 60
    X, y = [], []
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i-look_back:i])
        y.append(scaled_data[i, 0])  # Close price is target
    
    return np.array(X), np.array(y), scaler

async def train_and_evaluate(ticker, name):
    print(f"\nAnalyzing {name} ({ticker})")
    
    # Prepare data asynchronously
    X, y, scaler = await prepare_stock_data(ticker)
    
    # Check if we have enough data to split
    if len(X) < 5:  # Need at least a few samples
        print(f"Not enough data for {ticker} to perform train/test split")
        return 0, 0
    
    # Create train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create DataLoaders
    train_dataset = StockDataset(X_train, y_train)
    test_dataset = StockDataset(X_test, y_test)
    
    batch_size = min(32, len(X_train))  # Ensure batch size isn't larger than dataset
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    input_size = X_train.shape[2]
    model = LSTMModel(input_size=input_size, hidden_size=50, num_layers=1, output_size=1)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())
    
    # Training loop
    num_epochs = 50  # Reduced epochs for demonstration
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        batch_count = 0
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs.squeeze(), batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
        
        avg_loss = total_loss / max(1, batch_count)
        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')
    
    # Evaluation
    model.eval()
    with torch.no_grad():
        predictions = []
        actuals = []
        for batch_X, batch_y in test_loader:
            outputs = model(batch_X)
            predictions.extend(outputs.squeeze().tolist())
            actuals.extend(batch_y.tolist())
    
    # Convert to numpy arrays for inverse scaling
    predictions = np.array(predictions).reshape(-1, 1)
    actuals = np.array(actuals).reshape(-1, 1)
    
    # Verify we have predictions
    if len(predictions) == 0 or len(actuals) == 0:
        print(f"No predictions or actuals for {name}. Skipping evaluation.")
        return 0, 0
    
    # Create dummy arrays for inverse scaling
    dummy_predictions = np.hstack((predictions, np.zeros((predictions.shape[0], 6))))
    dummy_actuals = np.hstack((actuals, np.zeros((actuals.shape[0], 6))))
    
    predictions = scaler.inverse_transform(dummy_predictions)[:, 0]
    actuals = scaler.inverse_transform(dummy_actuals)[:, 0]
    
    # Calculate metrics
    mse = mean_squared_error(actuals, predictions)
    r2 = r2_score(actuals, predictions)
    
    print(f"\nResults for {name} ({ticker}):")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R-squared: {r2:.4f}")
    
    # Plot results
    plt.figure(figsize=(10, 5))
    plt.plot(actuals, label='Actual Prices')
    plt.plot(predictions, label='Predicted Prices')
    plt.title(f'{name} Stock Price Prediction')
    plt.xlabel('Time Steps')
    plt.ylabel('Price')
    plt.legend()
    plt.savefig(f"{ticker.replace('.', '_')}_prediction.png")
    plt.close()
    
    return mse, r2

async def main():
    """Main function using asyncio for concurrent stock data processing"""
    import asyncio
    
    # Limit to fewer stocks for demonstration
    selected_stocks = {
        'CBA.AX': 'Commonwealth Bank',
        'BHP.AX': 'BHP Group',
        'CSL.AX': 'CSL Limited'
    }
    
    # Process stocks concurrently
    async def process_stock(ticker, name):
        try:
            mse, r2 = await train_and_evaluate(ticker, name)
            return name, {'MSE': mse, 'R2': r2}
        except Exception as e:
            print(f"Error analyzing {name}: {e}")
            return name, {'MSE': 0, 'R2': 0, 'Error': str(e)}
    
    # Create tasks for all stocks
    tasks = [process_stock(ticker, name) for ticker, name in selected_stocks.items()]
    
    # Wait for all tasks to complete
    results_list = await asyncio.gather(*tasks)
    
    # Convert results to dictionary
    results = dict(results_list)
    
    # Print summary results
    print("\nSummary of Results:")
    for name, metrics in results.items():
        if 'Error' in metrics:
            print(f"{name}: Error - {metrics['Error']}")
        else:
            print(f"{name}: MSE = {metrics['MSE']:.4f}, R2 = {metrics['R2']:.4f}")

# After the main() function, add this function to display results beautifully
def display_results_pretty(results):
    print("\n" + "="*60)
    print("📈 STOCK SENTIMENT ANALYSIS RESULTS SUMMARY 📉")
    print("="*60)
    
    # Determine the best performing stock based on R2 score
    best_stock = max(results.items(), key=lambda x: x[1]['R2'] if 'R2' in x[1] else -1)
    
    for name, metrics in results.items():
        print("\n" + "🔹 " + name.upper() + " 🔹")
        
        if 'Error' in metrics:
            print(f"   ❌ Analysis failed: {metrics['Error']}")
            continue
            
        # MSE Interpretation
        mse_rating = "Excellent" if metrics['MSE'] < 1 else "Very Good" if metrics['MSE'] < 5 else "Good" if metrics['MSE'] < 10 else "Fair"
        print(f"   📊 Prediction Accuracy (MSE): {metrics['MSE']:.2f} → {mse_rating}")
        
        # R2 Interpretation
        r2_percentage = metrics['R2'] * 100
        r2_rating = "Excellent" if r2_percentage > 90 else "Very Good" if r2_percentage > 80 else "Good" if r2_percentage > 70 else "Fair"
        print(f"   📈 Model Fit (R²): {r2_percentage:.1f}% → {r2_rating}")
        
        # Special mention if this is the best performing stock
        if name == best_stock[0]:
            print("   🏆 BEST PERFORMING STOCK IN THIS ANALYSIS")
    
    print("\n" + "="*60)
    print("📌 LEGEND:")
    print("- MSE (Mean Squared Error): Lower values = better predictions")
    print("- R² Score: Percentage of price movement explained by the model")
    print("="*60)

# Add this function to generate trading signals
def generate_trading_signal(predicted_price_change, sentiment_score):
    """
    Generate trading signal based on:
    - Predicted price change (percentage)
    - Sentiment score (-1 to 1)
    
    Returns: Signal (BUY/SELL/HOLD), Confidence (High/Medium/Low)
    """
    # Determine price trend signal
    if predicted_price_change > 0.02:  # > 2% predicted increase
        price_signal = "BUY"
        price_confidence = "High"
    elif predicted_price_change > 0.005:  # 0.5%-2% increase
        price_signal = "BUY"
        price_confidence = "Medium"
    elif predicted_price_change < -0.02:  # > 2% predicted decrease
        price_signal = "SELL"
        price_confidence = "High"
    elif predicted_price_change < -0.005:  # 0.5%-2% decrease
        price_signal = "SELL"
        price_confidence = "Medium"
    else:
        price_signal = "HOLD"
        price_confidence = "Medium"
    
    # Determine sentiment signal
    if sentiment_score > 0.3:
        sentiment_signal = "BUY"
        sentiment_confidence = "High"
    elif sentiment_score > 0.1:
        sentiment_signal = "BUY"
        sentiment_confidence = "Medium"
    elif sentiment_score < -0.3:
        sentiment_signal = "SELL"
        sentiment_confidence = "High"
    elif sentiment_score < -0.1:
        sentiment_signal = "SELL"
        sentiment_confidence = "Medium"
    else:
        sentiment_signal = "HOLD"
        sentiment_confidence = "Medium"
    
    # Combine signals
    if price_signal == sentiment_signal:
        final_signal = price_signal
        final_confidence = "High"
    elif price_signal == "HOLD":
        final_signal = sentiment_signal
        final_confidence = sentiment_confidence
    elif sentiment_signal == "HOLD":
        final_signal = price_signal
        final_confidence = price_confidence
    else:  # Conflicting signals
        final_signal = "HOLD"
        final_confidence = "Low"
    
    return final_signal, final_confidence

# Modify the train_and_evaluate function to return signals
async def train_and_evaluate(ticker, name):
    print(f"\nAnalyzing {name} ({ticker})")
    
    # Prepare data asynchronously
    X, y, scaler = await prepare_stock_data(ticker)
    
    if len(X) < 5:
        print(f"Not enough data for {ticker} to perform train/test split")
        return None
    
    # Create train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create DataLoaders
    train_dataset = StockDataset(X_train, y_train)
    test_dataset = StockDataset(X_test, y_test)
    
    batch_size = min(32, len(X_train))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    input_size = X_train.shape[2]
    model = LSTMModel(input_size=input_size, hidden_size=50, num_layers=1, output_size=1)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())
    
    # Training loop
    num_epochs = 50
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        batch_count = 0
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs.squeeze(), batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
    
    # Get the latest sentiment from news
    headlines = await scrape_yahoo_finance()
    latest_sentiment = sum([get_sentiment(headline) for headline in headlines]) / len(headlines)
    
    # Make prediction for next day
    model.eval()
    with torch.no_grad():
        last_sequence = X[-1:]  # Get most recent sequence
        predicted_normalized = model(torch.FloatTensor(last_sequence))
        
        # Create dummy array for inverse scaling
        dummy_pred = np.zeros((1, 7))
        dummy_pred[0,0] = predicted_normalized.item()
        predicted_price = scaler.inverse_transform(dummy_pred)[0,0]
        
        # Get current price from last data point
        dummy_current = np.zeros((1, 7))
        dummy_current[0,0] = y[-1]
        current_price = scaler.inverse_transform(dummy_current)[0,0]
        
        # Calculate predicted change
        price_change = (predicted_price - current_price) / current_price
        
        # Generate trading signal
        signal, confidence = generate_trading_signal(price_change, latest_sentiment)
    
    return {
        'ticker': ticker,
        'name': name,
        'current_price': current_price,
        'predicted_price': predicted_price,
        'predicted_change': price_change,
        'sentiment': latest_sentiment,
        'signal': signal,
        'confidence': confidence
    }

# Update the display function
def display_results_pretty(results):
    print("\n" + "="*80)
    print("📈 STOCK TRADING SIGNALS BASED ON SENTIMENT & PRICE PREDICTION 📉")
    print("="*80)
    
    for stock in results:
        if stock is None:
            continue
            
        print(f"\n🔍 {stock['name']} ({stock['ticker']})")
        print(f"   💵 Current Price: ${stock['current_price']:.2f}")
        print(f"   🔮 Tomorrow's Predicted Price: ${stock['predicted_price']:.2f} ({stock['predicted_change']*100:.1f}%)")
        
        # Sentiment interpretation
        sentiment_emoji = "😊" if stock['sentiment'] > 0.1 else "😐" if stock['sentiment'] > -0.1 else "😞"
        sentiment_desc = "Positive" if stock['sentiment'] > 0.1 else "Neutral" if stock['sentiment'] > -0.1 else "Negative"
        print(f"   📰 News Sentiment: {sentiment_emoji} {sentiment_desc} ({stock['sentiment']:.2f})")
        
        # Signal with color
        if stock['signal'] == "BUY":
            signal_color = "\033[92m"  # Green
        elif stock['signal'] == "SELL":
            signal_color = "\033[91m"  # Red
        else:
            signal_color = "\033[93m"  # Yellow
        
        print(f"   🚦 Trading Signal: {signal_color}{stock['signal']}\033[0m (Confidence: {stock['confidence']})")
        
        # Recommendation reasoning
        print("\n   📝 Recommendation Reasoning:")
        if stock['signal'] == "BUY":
            if stock['predicted_change'] > 0.02:
                print("     - Strong predicted price increase (>2%)")
            else:
                print("     - Moderate predicted price increase")
            if stock['sentiment'] > 0.3:
                print("     - Very positive market sentiment")
            elif stock['sentiment'] > 0.1:
                print("     - Positive market sentiment")
        elif stock['signal'] == "SELL":
            if stock['predicted_change'] < -0.02:
                print("     - Strong predicted price decrease (>2%)")
            else:
                print("     - Moderate predicted price decrease")
            if stock['sentiment'] < -0.3:
                print("     - Very negative market sentiment")
            elif stock['sentiment'] < -0.1:
                print("     - Negative market sentiment")
        else:
            print("     - Neutral outlook based on mixed or inconclusive signals")
        
        print("   📈 Technical + 📰 Fundamental Analysis Combined")
    
    print("\n" + "="*80)
    print("ℹ️  Key:")
    print("- BUY/SELL signals combine price predictions with news sentiment analysis")
    print("- Confidence reflects agreement between technical and fundamental signals")
    print("="*80)

# Update main function
async def main():
    """Main function using asyncio for concurrent stock data processing"""
    selected_stocks = {
        'CBA.AX': 'Commonwealth Bank',
        'BHP.AX': 'BHP Group',
        'CSL.AX': 'CSL Limited'
    }
    
    # Process stocks concurrently
    tasks = [train_and_evaluate(ticker, name) for ticker, name in selected_stocks.items()]
    results = await asyncio.gather(*tasks)
    
    # Display pretty results
    display_results_pretty(results)

if __name__ == "__main__":
    asyncio.run(main())