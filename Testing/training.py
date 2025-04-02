import os
import warnings
import pandas as pd
import numpy as np
import sklearn
import xgboost
import tensorflow as tf
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import EarlyStopping
from joblib import Memory

# Configuration
warnings.filterwarnings("ignore", category=UserWarning)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU
os.makedirs("cached_models", exist_ok=True)
os.makedirs("trained_models", exist_ok=True)

# Cache setup
memory = Memory("cached_models", verbose=0)

def prepare_data(selected_data, coin_index=0):
    """Prepare data with lag features"""
    selected_data = selected_data.copy()
    coin_name = selected_data.columns[coin_index]
    
    for lag in range(1, 4):
        selected_data[f'{coin_name}_lag_{lag}'] = selected_data[coin_name].shift(lag)
    
    selected_data.dropna(inplace=True)
    
    features = [f'{coin_name}_lag_{lag}' for lag in range(1, 4)]
    X = selected_data[features]
    y = selected_data[coin_name]
    
    return train_test_split(X, y, test_size=0.2, random_state=42)

@memory.cache
def train_gradient_boosting(X_train, y_train):
    """Train Gradient Boosting model with caching"""
    params = {
        'n_estimators': [50, 100],
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 5],
        'min_samples_leaf': [1, 2],
        'subsample': [0.8, 0.9]
    }
    gb = GradientBoostingRegressor(random_state=42)
    model = GridSearchCV(gb, params, cv=5, scoring='neg_mean_squared_error', verbose=1)
    model.fit(X_train, y_train)
    return model.best_estimator_

@memory.cache
def train_svr(X_train, y_train):
    """Train SVR model with caching"""
    params = {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto']
    }
    svr = SVR()
    model = GridSearchCV(svr, params, cv=5, scoring='neg_mean_squared_error', verbose=1)
    model.fit(X_train, y_train)
    return model.best_estimator_

@memory.cache
def train_xgboost(X_train, y_train):
    """Train XGBoost model with caching"""
    params = {
        'n_estimators': [50, 100],
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 5],
        'subsample': [0.8, 0.9],
        'colsample_bytree': [0.8, 0.9]
    }
    xgb = XGBRegressor(random_state=42, enable_categorical=True)  # Updated for xgboost 3.0.0
    model = GridSearchCV(xgb, params, cv=5, scoring='neg_mean_squared_error', verbose=1)
    model.fit(X_train, y_train)
    return model.best_estimator_

def train_lstm(X_train, y_train):
    """Train LSTM model compatible with TensorFlow 2.19.0"""
    model = Sequential([
        LSTM(64, input_shape=(X_train.shape[1], 1), return_sequences=False),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    X_train_reshaped = X_train.to_numpy().reshape(X_train.shape[0], X_train.shape[1], 1)
    
    history = model.fit(
        X_train_reshaped, y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=1
    )
    
    return model

def save_model(model, model_name, coin_index=1):
    """Save trained model to file with version info"""
    model_dir = f"trained_models/Model_SELECTED_COIN_{coin_index}"
    os.makedirs(model_dir, exist_ok=True)
    
    # Save version information
    with open(f"{model_dir}/requirements.txt", "w") as f:
        f.write(f"scikit-learn=={sklearn.__version__}\n")
        f.write(f"xgboost=={xgboost.__version__}\n")
        f.write(f"tensorflow=={tf.__version__}\n")
        f.write(f"joblib=={joblib.__version__}\n")
    
    if model_name == 'LSTM':
        model_path = f"{model_dir}/lstm_model.keras"
        model.save(model_path)
    else:
        model_path = f"{model_dir}/{model_name.lower().replace(' ', '_')}_model.pkl"
        joblib.dump(model, model_path, compress=3)  # Using compress=3 for smaller files
    
    # Save training metadata
    metadata = {
        'training_date': pd.Timestamp.now().isoformat(),
        'input_shape': model.n_features_in_ if hasattr(model, 'n_features_in_') else X_train.shape[1],
        'model_type': model_name
    }
    joblib.dump(metadata, f"{model_dir}/{model_name.lower().replace(' ', '_')}_metadata.pkl")
    
    return model_path

def train_all_models(selected_data, coin_index=0):
    """Train all models for a specific coin with progress tracking"""
    try:
        X_train, X_test, y_train, y_test = prepare_data(selected_data, coin_index)
        
        models = {
            'Gradient Boosting': train_gradient_boosting(X_train, y_train),
            'SVR': train_svr(X_train, y_train),
            'XGBoost': train_xgboost(X_train, y_train),
            'LSTM': train_lstm(X_train, y_train)
        }
        
        saved_paths = {}
        for name, model in models.items():
            saved_paths[name] = save_model(model, name, coin_index + 1)
        
        # Evaluate and save performance metrics
        performance = {}
        for name, model in models.items():
            if name == 'LSTM':
                X_test_reshaped = X_test.to_numpy().reshape(X_test.shape[0], X_test.shape[1], 1)
                predictions = model.predict(X_test_reshaped).flatten()
            else:
                predictions = model.predict(X_test)
            
            performance[name] = {
                'MAE': mean_absolute_error(y_test, predictions),
                'MSE': mean_squared_error(y_test, predictions),
                'R2': r2_score(y_test, predictions)
            }
        
        joblib.dump(performance, f"trained_models/Model_SELECTED_COIN_{coin_index+1}/performance_metrics.pkl")
        
        return saved_paths
    
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise

if __name__ == "__main__":
    # For standalone training
    selected_data = pd.read_csv("Selected_coins.csv", index_col='Date')
    print("Training models...")
    
    # Train models for each coin
    for coin_idx in range(selected_data.shape[1]):
        print(f"\nTraining models for coin {coin_idx+1} of {selected_data.shape[1]}")
        train_all_models(selected_data, coin_idx)
    
    print("\nTraining completed successfully!")