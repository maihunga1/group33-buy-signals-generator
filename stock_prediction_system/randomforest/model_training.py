import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from imblearn.pipeline import make_pipeline as make_imb_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from datetime import datetime
from feature_engineering import get_feature_columns
from utils import get_model_path
from config import RANDOM_STATE, RETURN_THRESHOLD

def train_model(data):
    """Train an improved model with hyperparameter tuning and class imbalance handling"""
    features = get_feature_columns()
    X = data[features]
    y = data['Buy_Signal']
    
    # Check class distribution
    class_counts = y.value_counts()
    print("\n🔢 Class distribution:")
    print(f"Buy signals (1): {class_counts.get(1, 0)} ({class_counts.get(1, 0)/len(y)*100:.1f}%)")
    print(f"No-buy signals (0): {class_counts.get(0, 0)} ({class_counts.get(0, 0)/len(y)*100:.1f}%)")
    
    # Split data - using stratified sampling to maintain class proportions
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    
    # Create preprocessing pipeline
    numeric_transformer = make_pipeline(
        SimpleImputer(strategy='median'),  # Handle missing values
        StandardScaler()  # Scale features
    )
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, features)
        ])
    
    # Handle class imbalance with SMOTE in a pipeline
    if class_counts.get(1, 0) >= 5:  # Need some minimum samples for SMOTE
        print("🔄 Applying SMOTE to balance classes...")
        model = make_imb_pipeline(
            preprocessor,
            SMOTE(random_state=RANDOM_STATE),
            RandomForestClassifier(random_state=RANDOM_STATE)
        )
    else:
        print("⚠️ Not enough minority samples for SMOTE, using original data")
        model = make_pipeline(
            preprocessor,
            RandomForestClassifier(random_state=RANDOM_STATE)
        )
    
    # Hyperparameter tuning
    print("\n🔍 Performing hyperparameter tuning...")
    param_grid = {
        'randomforestclassifier__n_estimators': [100, 200],
        'randomforestclassifier__max_depth': [None, 10, 20],
        'randomforestclassifier__min_samples_split': [2, 5],
        'randomforestclassifier__min_samples_leaf': [1, 2],
        'randomforestclassifier__class_weight': [None, 'balanced']
    }
    
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=3,
        scoring='f1',
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    best_model = grid_search.best_estimator_
    
    # Evaluate the model
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]  # Probabilities for class 1
    
    print("\n📊 Model Evaluation:")
    print(classification_report(y_test, y_pred))
    
    try:
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        print(f"ROC AUC Score: {roc_auc:.3f}")
    except:
        print("Could not calculate ROC AUC Score")
    
    # Save model and preprocessing steps
    model_data = {
        'model': best_model,
        'features': features,
        'y_test': y_test,
        'y_pred': y_pred,
        'threshold': RETURN_THRESHOLD,
        'training_date': datetime.now().strftime('%Y-%m-%d'),
    }
    
    joblib.dump(model_data, get_model_path('stock_prediction_model.pkl'))
    print("✅ Model saved as 'stock_prediction_model.pkl'")
    
    return model_data