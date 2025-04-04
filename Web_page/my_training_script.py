import warnings
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense
import tensorflow as tf
import joblib

# Suppressing warnings
warnings.filterwarnings("ignore", category=UserWarning)

def train_models(selected_data):
    # Making a copy of the slice to ensure it's a separate object
    selected_data = selected_data.copy()

    for lag in range(1, 4):  # Adding lagged features for 1 to 3 days
        selected_data.loc[:, f'{selected_data.columns[0]}_lag_{lag}'] = selected_data[selected_data.columns[0]].shift(lag)

    # Dropping rows with NaN values created due to shifting
    selected_data.dropna(inplace=True)

    # Features will be the lagged values, and the target will be the current price of the first coin
    features = [f'{selected_data.columns[0]}_lag_{lag}' for lag in range(1, 4)]
    X = selected_data[features]
    y = selected_data[selected_data.columns[0]]

    # Splitting the dataset into training and testing sets
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the models
    models = {
        'Gradient Boosting': GradientBoostingRegressor(),
        'SVR': SVR(),
        'XGBoost': XGBRegressor(),
        'LSTM': Sequential([LSTM(units=50, input_shape=(X_train.shape[1], 1)), Dense(units=1)])
    }

    for model_name, model in models.items():
        if model_name == 'Gradient Boosting':
            params = {
                'n_estimators': [50, 100],
                'learning_rate': [0.01, 0.1],
                'max_depth': [3, 5],
                'min_samples_leaf': [1, 2],
                'subsample': [0.8, 0.9]
            }
            model = GridSearchCV(estimator=model, param_grid=params, cv=5, scoring='neg_mean_squared_error')
            model.fit(X_train, y_train)
            model = model.best_estimator_

        elif model_name == 'SVR':
            params = {
                'C': [0.1, 1],
                'kernel': ['linear', 'rbf'],
                'gamma': ['scale', 'auto']
            }
            model = GridSearchCV(estimator=model, param_grid=params, cv=5, scoring='neg_mean_squared_error')
            model.fit(X_train, y_train)
            model = model.best_estimator_

        elif model_name == 'XGBoost':
            params = {
                'n_estimators': [50, 100],
                'learning_rate': [0.01, 0.1],
                'max_depth': [3, 5],
                'subsample': [0.8, 0.9]
            }
            model = GridSearchCV(estimator=model, param_grid=params, cv=5, scoring='neg_mean_squared_error')
            model.fit(X_train, y_train)
            model = model.best_estimator_

        # For LSTM model
        elif model_name == 'LSTM':
            model.add(Dense(units=1, name='output'))  # Add output layer with a unique name
            model.compile(optimizer='adam', loss='mean_squared_error')
            X_train_array = X_train.to_numpy().reshape(X_train.shape[0], X_train.shape[1], 1)
            model.fit(X_train_array, y_train, epochs=100, batch_size=32, verbose=0)

            # Save the trained LSTM model
            model_filename = f"Model_SELECTED_COIN_1/{model_name.lower().replace(' ', '_')}_model"
            model.save(model_filename)
            print(f"Trained {model_name} model saved as {model_filename}")

            
        # Save the trained model for other models
        else:
            model_filename = f"Model_SELECTED_COIN_1/{model_name.lower().replace(' ', '_')}_model.pkl"
            os.makedirs(os.path.dirname(model_filename), exist_ok=True)
            joblib.dump(model, model_filename)
            print(f"Trained {model_name} model saved as {model_filename}")

# Load the selected data from a CSV file
selected_data = pd.read_csv("Selected_coins.csv", index_col='Date')

# Usage
train_models(selected_data)



# Function to train models in the background when the app starts
def train_models_background(selected_data):
    st.write("Training models in the background...")
    train_models(selected_data)
    st.write("Training completed.")
    
    
    

# Train models in the background when the app starts
train_models_background(selected_data)



from my_training_script import train_models
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
