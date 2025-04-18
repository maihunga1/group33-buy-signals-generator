import pandas as pd
import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from fetch_data import fetch_and_calculate
import os
import pandas as pd
import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from fetch_data import fetch_and_calculate
import numpy as np

# List of stock tickers (ASX 200 or any stock universe you're interested in)
TICKERS = ['BHP.AX', 'CBA.AX', 'NAB.AX', 'CSL.AX', 'WBC.AX']

# Create the 'xgboost' directory if it doesn't exist
folder_path = 'data'
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
# Fetch the data and save it as a single CSV
combined_data = fetch_and_calculate(TICKERS)

# Rest of your code...
# List of stock tickers (ASX 200 or any stock universe you're interested in)
TICKERS = ['BHP.AX', 'CBA.AX', 'NAB.AX', 'CSL.AX', 'WBC.AX']

# Create the 'xgboost' directory if it doesn't exist
folder_path = 'data'
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
# Fetch the data and save it as a single CSV
combined_data = fetch_and_calculate(TICKERS)


def create_labels(df, threshold=0.05, remove_uncertain=True):
    df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    # Calculate future close price (next row within the same ticker)
    df['Future_Close'] = df.groupby("Ticker")['Close'].shift(-1)

    # Calculate percentage change
    df['Pct_Change'] = (df['Future_Close'] / df['Close']) - 1

    # Apply threshold logic
    df['Label'] = df['Pct_Change'].apply(
        lambda x: 1 if x >= threshold else (0 if x <= -threshold else np.nan)
    )

    # Optionally remove uncertain samples
    if remove_uncertain:
        df = df.dropna(subset=["Label"])

    df['Label'] = df['Label'].astype(int)

    return df


# Apply labels to the combined data
combined_data = create_labels(combined_data)

# Preparing features and labels for training
features = ['Price_Change', 'SMA_20', 'RSI', 'ATR', 'Price_MA_Ratio']
X = combined_data[features]
y = combined_data['Label']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost model on the combined data
model = xgb.XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='mlogloss')
model.fit(X_train, y_train)

# Save the trained model to the 'xgboost' folder
model_filename = os.path.join(folder_path, 'combined_stock_model.pkl')
joblib.dump(model, model_filename)

# Evaluate the model and save the results
y_pred = model.predict(X_test)
report = classification_report(y_test, y_pred)

# Save the classification report to a text file
report_filename = os.path.join(folder_path, 'combined_classification_report.txt')
with open(report_filename, 'w') as file:
    file.write(report)

print(f"Model performance saved to: {report_filename}")
print(f"Model saved to: {model_filename}")
