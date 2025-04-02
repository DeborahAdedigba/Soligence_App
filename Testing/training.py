import os
import warnings
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import EarlyStopping
import tensorflow as tf
import joblib
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
    gb = GradientBoostingRegressor()
    model = GridSearchCV(gb, params, cv=5, scoring='neg_mean_squared_error')
    model.fit(X_train, y_train)
    return model.best_estimator_

@memory.cache
def train_svr(X_train, y_train):
    """Train SVR model with caching"""
    params = {
        'C': [0.1, 1],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto']
    }
    svr = SVR()
    model = GridSearchCV(svr, params, cv=5, scoring='neg_mean_squared_error')
    model.fit(X_train, y_train)
    return model.best_estimator_

@memory.cache
def train_xgboost(X_train, y_train):
    """Train XGBoost model with caching"""
    params = {
        'n_estimators': [50, 100],
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 5],
        'subsample': [0.8, 0.9]
    }
    xgb = XGBRegressor()
    model = GridSearchCV(xgb, params, cv=5, scoring='neg_mean_squared_error')
    model.fit(X_train, y_train)
    return model.best_estimator_

def train_lstm(X_train, y_train):
    """Train LSTM model"""
    model = Sequential([
        LSTM(32, input_shape=(X_train.shape[1], 1), return_sequences=False),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    
    early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
    
    X_train_reshaped = X_train.to_numpy().reshape(X_train.shape[0], X_train.shape[1], 1)
    
    model.fit(
        X_train_reshaped, y_train,
        epochs=30,
        batch_size=32,
        callbacks=[early_stop],
        verbose=0
    )
    return model

# In your training.py, modify the save_model function to include version info
def save_model(model, model_name, coin_index=1):
    """Save trained model to file with version info"""
    model_dir = f"trained_models/Model_SELECTED_COIN_{coin_index}"
    os.makedirs(model_dir, exist_ok=True)
    
    # Save version information
    with open(f"{model_dir}/requirements.txt", "w") as f:
        f.write(f"scikit-learn=={sklearn.__version__}\n")
        f.write(f"xgboost=={xgboost.__version__}\n")
        f.write(f"tensorflow=={tf.__version__}\n")
    
    if model_name == 'LSTM':
        model_path = f"{model_dir}/lstm_model.keras"
        model.save(model_path)
    else:
        model_path = f"{model_dir}/{model_name.lower().replace(' ', '_')}_model.pkl"
        joblib.dump(model, model_path)
    
    return model_path

def train_all_models(selected_data, coin_index=0):
    """Train all models for a specific coin"""
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
    
    return saved_paths

if __name__ == "__main__":
    # For standalone training
    selected_data = pd.read_csv("Selected_coins.csv", index_col='Date')
    print("Training models...")
    train_all_models(selected_data)
    print("Training completed!")

# import pickle
# import os
# import pandas as pd
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.ensemble import GradientBoostingRegressor
# from sklearn.svm import SVR
# from xgboost import XGBRegressor
# from keras.models import Sequential
# from keras.layers import LSTM, Dense
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# # Load the selected data from a CSV file
# selected_data = pd.read_csv("Selected_coins.csv", index_col='Date')

# def train_and_save_models(selected_data, coin_number):
#     selected_data = selected_data.copy()

#     for lag in range(1, 4):
#         selected_data.loc[:, f'{selected_data.columns[coin_number]}_lag_{lag}'] = selected_data[selected_data.columns[coin_number]].shift(lag)

#     selected_data.dropna(inplace=True)

#     features = [f'{selected_data.columns[coin_number]}_lag_{lag}' for lag in range(1, 4)]
#     X = selected_data[features]
#     y = selected_data[selected_data.columns[coin_number]]

#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#     models = {
#         'Gradient Boosting': GradientBoostingRegressor(),
#         'SVR': SVR(),
#         'XGBoost': XGBRegressor(),
#         'LSTM': Sequential([LSTM(units=50, input_shape=(X_train.shape[1], 1))])  # Define LSTM model without Dense layer
#     }

#     # Hyperparameter grid for grid search
#     param_grid = {
#         'Gradient Boosting': {
#             'n_estimators': [50, 100],
#             'learning_rate': [0.1, 0.01],
#             'max_depth': [3, 5],
#             'min_samples_leaf': [1, 2],
#             'subsample': [0.8, 0.9]
#         },
#         'SVR': {'C': [1, 10], 'gamma': ['scale', 'auto']},
#         'XGBoost': {
#             'n_estimators': [50, 100],
#             'learning_rate': [0.1, 0.01],
#             'max_depth': [3, 5],
#             'subsample': [0.8, 0.9]
#         },
#     }

#     for model_name, model in models.items():
#         if model_name == 'LSTM':
#             model.add(Dense(units=1))  # Add Dense layer to LSTM model
#             model.compile(optimizer='adam', loss='mean_squared_error')
#             X_train_array = X_train.to_numpy().reshape(X_train.shape[0], X_train.shape[1], 1)
#             model.fit(X_train_array, y_train, epochs=100, batch_size=32, verbose=0)

#             # Save the trained LSTM model
#             model_filename = f"Model_SELECTED_COIN_{coin_number+1}/lstm_model.h5"
#             os.makedirs(os.path.dirname(model_filename), exist_ok=True)
#             model.save(model_filename)
#             print(f"Trained {model_name} model saved as {model_filename}")
#             continue  # Skip the rest of the loop for LSTM

#         # Perform grid search for hyperparameter tuning
#         params = param_grid[model_name]
#         model = GridSearchCV(estimator=model, param_grid=params, cv=5, scoring='neg_mean_squared_error')
#         model.fit(X_train, y_train)
#         model = model.best_estimator_

#         # Train other models
#         model.fit(X_train, y_train)

#         # Save the trained model
#         model_filename = f"Model_SELECTED_COIN_{coin_number+1}/{model_name.lower().replace(' ', '_')}_model.pkl"
#         os.makedirs(os.path.dirname(model_filename), exist_ok=True)
#         with open(model_filename, 'wb') as f:
#             pickle.dump(model, f)
#         print(f"Trained {model_name} model saved as {model_filename}")

# # Call the function to train and save all models for each coin
# for coin_number in range(selected_data.shape[1]):
#     train_and_save_models(selected_data, coin_number)

