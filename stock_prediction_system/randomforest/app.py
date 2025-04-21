from flask import Flask, jsonify, request
import pandas as pd
import joblib
from signal_generator import generate_predictions

# Initialize Flask app
app = Flask(__name__)

# Load model once
model_data = joblib.load('models/stock_prediction_model.pkl')

# Load data once (or you can load it dynamically based on request)
data = pd.read_csv('data/processed/processed_data.csv')

@app.route('/predict', methods=['GET'])
def predict():
    # Get ticker from request args
    ticker = request.args.get('ticker')
    
    # Filter data for the specific ticker
    ticker_data = data[data['Ticker'] == ticker]
    
    if ticker_data.empty:
        return jsonify({'error': 'Ticker not found'}), 404
    
    # Generate predictions
    predictions = generate_predictions(model_data, ticker_data)
    
    # Prepare response
    response = []
    for _, row in predictions.iterrows():
        response.append({
            'Ticker': row['Ticker'],
            'Recommendation': row['Recommendation'],
            'Rationale': row['Rationale'],
            'Top_3_Features': row['Top_3_Features']
        })
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)