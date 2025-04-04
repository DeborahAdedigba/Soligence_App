# importing necessary modules
import os
import warnings
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from ta.trend import SMAIndicator
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import mplfinance as mpf
import matplotlib.dates as mdates
import feedparser
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
import tensorflow as tf
import itertools
from scipy.stats import gaussian_kde
import threading
import time
import pickle
import yfinance as yf
from plotly.subplots import make_subplots
from joblib import Memory
import pkg_resources
from keras.callbacks import EarlyStopping
import logging
from logging.handlers import RotatingFileHandler
import sklearn
from sklearn import __version__ as sklearn_version

warnings.filterwarnings("ignore", category=UserWarning)

# Configuration for training
os.makedirs("cached_models", exist_ok=True)
os.makedirs("trained_models", exist_ok=True)
memory = Memory("cached_models", verbose=0)


def initialize_session_state():
    """Initialize all required session state variables"""
    required_vars = {
        'models_trained': False,
        'model_paths': {},
        'training_progress': {},
        'training_thread': None,
        'training_started': False,
        'last_update': time.time()
    }
    
    for var, default in required_vars.items():
        if var not in st.session_state:
            st.session_state[var] = default

# Initialize session state
def initialize_session_state():
    """Initialize all required session state variables"""
    if 'models_trained' not in st.session_state:
        st.session_state.models_trained = False
    if 'model_paths' not in st.session_state:
        st.session_state.model_paths = {}
    if 'training_progress' not in st.session_state:
        st.session_state.training_progress = {}  # For per-coin progress
    if 'overall_progress' not in st.session_state:
        st.session_state.overall_progress = 0  # For tracking total models completed
    if 'training_thread' not in st.session_state:
        st.session_state.training_thread = None
    if 'training_started' not in st.session_state:
        st.session_state.training_started = False
    if 'last_update' not in st.session_state:
        st.session_state.last_update = time.time()

# Version checking
def check_versions():
    required = {
        'scikit-learn': '1.6.1',
        'xgboost': '3.0.0',
        'tensorflow': '2.19.0',
        'joblib': '1.4.2'
    }
    
    for pkg, req_ver in required.items():
        try:
            current_ver = globals()[f"{pkg.replace('-', '_')}_version"]
            if current_ver != req_ver:
                logging.warning(f"Version mismatch for {pkg}: installed {current_ver}, expected {req_ver}")
        except Exception as e:
            logging.error(f"Version check failed for {pkg}: {str(e)}")

# Training functions
# def prepare_data(selected_data, coin_index=0):
#     """Prepare data with lag features"""
#     selected_data = selected_data.copy()
#     coin_name = selected_data.columns[coin_index]
    
#     for lag in range(1, 4):
#         selected_data[f'{coin_name}_lag_{lag}'] = selected_data[coin_name].shift(lag)
    
#     selected_data.dropna(inplace=True)
    
#     features = [f'{coin_name}_lag_{lag}' for lag in range(1, 4)]
#     X = selected_data[features]
#     y = selected_data[coin_name]
    
#     return train_test_split(X, y, test_size=0.2, random_state=42)

def prepare_data(selected_data, coin_index=0, for_lstm=False):
    """Prepare data with lag features"""
    selected_data = selected_data.copy()
    coin_name = selected_data.columns[coin_index]
    
    for lag in range(1, 4):
        selected_data[f'{coin_name}_lag_{lag}'] = selected_data[coin_name].shift(lag)
    
    selected_data.dropna(inplace=True)
    
    features = [f'{coin_name}_lag_{lag}' for lag in range(1, 4)]
    X = selected_data[features]
    y = selected_data[coin_name]
    
    # Convert to numpy arrays only if requested (for LSTM)
    if for_lstm:
        X = X.values.astype(np.float32)
        y = y.values.astype(np.float32)
    
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_gradient_boosting(X_train, y_train):
    """Train Gradient Boosting model with simplified parameters for faster training"""
    params = {
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 3,
        'min_samples_leaf': 2,
        'subsample': 0.8,
        'random_state': 42
    }
    gb = GradientBoostingRegressor(**params)
    gb.fit(X_train, y_train)
    return gb

def train_svr(X_train, y_train):
    """Train SVR model with simplified parameters"""
    params = {
        'C': 1.0,
        'kernel': 'rbf',
        'gamma': 'scale'
    }
    svr = SVR(**params)
    svr.fit(X_train, y_train)
    return svr

def train_xgboost(X_train, y_train):
    """Train XGBoost model with simplified parameters"""
    params = {
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 3,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42
    }
    xgb = XGBRegressor(**params)
    xgb.fit(X_train, y_train)
    return xgb


# Add these imports at the TOP of your script (with other imports)
import sklearn
from sklearn import __version__ as sklearn_version
import xgboost
from xgboost import __version__ as xgboost_version
import tensorflow as tf
import joblib
from joblib import __version__ as joblib_version

# Then modify your save_model function to this robust version:
def save_model(model, model_name, coin_index=1, input_shape=None):
    """Save trained model to file with version info"""
    try:
        model_dir = f"trained_models/Model_SELECTED_COIN_{coin_index}"
        os.makedirs(model_dir, exist_ok=True)
        
        # Write requirements file with fallback version handling
        requirements_file = os.path.join(model_dir, "requirements.txt")
        with open(requirements_file, "w") as f:
            # Get versions with fallbacks
            sklearn_ver = getattr(sklearn, '__version__', 'unknown')
            xgboost_ver = getattr(xgboost, '__version__', 'unknown')
            tf_ver = getattr(tf, '__version__', 'unknown')
            joblib_ver = getattr(joblib, '__version__', 'unknown')
            
            f.write(f"scikit-learn=={sklearn_ver}\n")
            f.write(f"xgboost=={xgboost_ver}\n") 
            f.write(f"tensorflow=={tf_ver}\n")
            f.write(f"joblib=={joblib_ver}\n")
        
        # Save the model based on type
        if model_name == 'LSTM':
            model_path = os.path.join(model_dir, "lstm_model.keras")
            model.save(model_path)
        else:
            model_path = os.path.join(model_dir, f"{model_name.lower().replace(' ', '_')}_model.pkl")
            joblib.dump(model, model_path, compress=3)
        
        # Save metadata
        metadata = {
            'training_date': pd.Timestamp.now().isoformat(),
            'input_shape': input_shape,
            'model_type': model_name,
            'versions': {
                'scikit-learn': sklearn_ver,
                'xgboost': xgboost_ver,
                'tensorflow': tf_ver,
                'joblib': joblib_ver
            }
        }
        metadata_path = os.path.join(model_dir, f"{model_name.lower().replace(' ', '_')}_metadata.pkl")
        joblib.dump(metadata, metadata_path)
        
        return model_dir
        
    except Exception as e:
        logging.error(f"Error saving model {model_name}: {str(e)}")
        raise

# Configure logging
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            RotatingFileHandler('training.log', maxBytes=1e6, backupCount=3),
            logging.StreamHandler()
        ]
    )
    logging.info("Logging system initialized")

setup_logging()

# Modify the train_models_for_coin function to add logging
def train_models_for_coin(selected_data, coin_index):
    """Train all models for a specific coin with progress tracking"""
    # Initialize session state for this thread
    initialize_session_state()
    
    coin_name = selected_data.columns[coin_index]
    logging.info(f"Starting training for {coin_name}")
    
    try:
        # Thread-safe session state updates
        with threading.Lock():
            # Ensure training_progress is a dictionary
            if not isinstance(st.session_state.training_progress, dict):
                st.session_state.training_progress = {}
                
            st.session_state.training_progress[coin_name] = {
                'status': 'In Progress',
                'current_model': None,
                'progress': 0,
                'message': ''
            }
            st.session_state.last_update = time.time()

        models = {}
        
        # ===== GRADIENT BOOSTING =====
        logging.info(f"Training Gradient Boosting for {coin_name}")
        with threading.Lock():
            st.session_state.training_progress[coin_name]['current_model'] = 'Gradient Boosting'
            st.session_state.training_progress[coin_name]['message'] = 'Training Gradient Boosting...'
            st.session_state.last_update = time.time()
        
        # Prepare data (DataFrame format)
        X_train, X_test, y_train, y_test = prepare_data(selected_data, coin_index, for_lstm=False)
        models['Gradient Boosting'] = train_gradient_boosting(X_train, y_train)
        
        with threading.Lock():
            st.session_state.training_progress[coin_name]['progress'] = 25
            st.session_state.training_progress[coin_name]['message'] = 'Gradient Boosting completed'
            st.session_state.last_update = time.time()
        
        # ===== SVR =====
        logging.info(f"Training SVR for {coin_name}")
        with threading.Lock():
            st.session_state.training_progress[coin_name]['current_model'] = 'SVR'
            st.session_state.training_progress[coin_name]['message'] = 'Training SVR...'
            st.session_state.last_update = time.time()
        
        # Reuse same DataFrame data
        models['SVR'] = train_svr(X_train, y_train)
        
        with threading.Lock():
            st.session_state.training_progress[coin_name]['progress'] = 50
            st.session_state.training_progress[coin_name]['message'] = 'SVR completed'
            st.session_state.last_update = time.time()
        
        # ===== XGBOOST =====
        logging.info(f"Training XGBoost for {coin_name}")
        with threading.Lock():
            st.session_state.training_progress[coin_name]['current_model'] = 'XGBoost'
            st.session_state.training_progress[coin_name]['message'] = 'Training XGBoost...'
            st.session_state.last_update = time.time()
        
        # Reuse same DataFrame data
        models['XGBoost'] = train_xgboost(X_train, y_train)
        
        with threading.Lock():
            st.session_state.training_progress[coin_name]['progress'] = 75
            st.session_state.training_progress[coin_name]['message'] = 'XGBoost completed'
            st.session_state.last_update = time.time()
        
        # ===== LSTM =====
        logging.info(f"Training LSTM for {coin_name} (this may take a while)")
        with threading.Lock():
            st.session_state.training_progress[coin_name]['current_model'] = 'LSTM'
            st.session_state.training_progress[coin_name]['message'] = 'Training LSTM (this may take a few minutes)...'
            st.session_state.last_update = time.time()
        
        # Prepare fresh data in NumPy format
        X_train_lstm, _, y_train_lstm, _ = prepare_data(selected_data, coin_index, for_lstm=True)
        models['LSTM'] = train_lstm(X_train_lstm, y_train_lstm)
        
        with threading.Lock():
            st.session_state.training_progress[coin_name]['progress'] = 100
            st.session_state.training_progress[coin_name]['message'] = 'LSTM completed'
            st.session_state.last_update = time.time()
        
        # Save models
        logging.info(f"Saving models for {coin_name}")
        saved_paths = {}
        for name, model in models.items():
            model_dir = save_model(model, name, coin_index + 1, X_train.shape[1])
            saved_paths[name] = model_dir
        
        with threading.Lock():
            if not hasattr(st.session_state, 'model_paths'):
                st.session_state.model_paths = {}
            st.session_state.model_paths.update(saved_paths)
            st.session_state.last_update = time.time()
        
        logging.info(f"Successfully completed training for {coin_name}")
        return models
        
    except Exception as e:
        error_msg = f"Error training {coin_name}: {str(e)}"
        logging.error(error_msg, exc_info=True)
        with threading.Lock():
            if coin_name in st.session_state.training_progress:
                st.session_state.training_progress[coin_name]['status'] = f'Failed: {str(e)}'
                st.session_state.training_progress[coin_name]['message'] = error_msg
            st.session_state.last_update = time.time()
        raise e


# def train_lstm(X_train, y_train):
#     """Train LSTM model (requires numpy arrays)"""
#     logging.info("Initializing LSTM model")

#     tf.keras.backend.clear_session()
    
#     model = tf.keras.Sequential([
#         tf.keras.layers.LSTM(32, input_shape=(X_train.shape[1], 1)),
#         tf.keras.layers.Dense(16, activation='relu'),
#         tf.keras.layers.Dense(1)
#     ])
    
#     model.compile(optimizer='adam', loss='mse')
    
#     # Ensure data is numpy array
#     X_array = np.array(X_train, dtype=np.float32).reshape(X_train.shape[0], X_train.shape[1], 1)
#     y_array = np.array(y_train, dtype=np.float32)
    
#     model.fit(X_array, y_array, epochs=50, batch_size=32, verbose=1)


#     logging.info("Starting LSTM training")
#     history = model.fit(
#         X_train_reshaped, y_train,
#         epochs=50,
#         batch_size=32,
#         validation_split=0.2,
#         callbacks=[early_stop],
#         verbose=1
#     )
#     logging.info("Completed LSTM training")
    
#     return model
def train_lstm(X_train, y_train):
    """Train LSTM model (requires numpy arrays)"""
    logging.info("Initializing LSTM model training")
    
    try:
        # Clear any existing session
        tf.keras.backend.clear_session()
        logging.info("Cleared previous Keras session")

        # Define early stopping callback
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        )
        logging.info("Configured early stopping callback")

        # Build model
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(32, input_shape=(X_train.shape[1], 1)),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
        logging.info("Created LSTM model architecture")

        # Compile model
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        logging.info("Compiled LSTM model")

        # Prepare data
        logging.info("Preparing training data")
        X_train_reshaped = np.array(X_train, dtype=np.float32).reshape(
            X_train.shape[0], X_train.shape[1], 1
        )
        y_train_array = np.array(y_train, dtype=np.float32)
        logging.info(f"Training data shape: {X_train_reshaped.shape}")
        logging.info(f"Target data shape: {y_train_array.shape}")

        # Train model
        logging.info("Starting LSTM training")
        history = model.fit(
            X_train_reshaped, 
            y_train_array,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            callbacks=[early_stop],
            verbose=1
        )
        
        # Log training results
        final_epoch = len(history.history['loss'])
        final_loss = history.history['loss'][-1]
        final_val_loss = history.history.get('val_loss', [None])[-1]
        
        logging.info(f"Completed LSTM training after {final_epoch} epochs")
        logging.info(f"Final training loss: {final_loss:.4f}")
        if final_val_loss is not None:
            logging.info(f"Final validation loss: {final_val_loss:.4f}")
        
        return model

    except Exception as e:
        logging.error(f"Error during LSTM training: {str(e)}", exc_info=True)
        return None

# Modify the train_all_models_background function
def train_all_models_background(selected_data):
    """Train models for all coins in a background thread with proper progress tracking"""
    def initialize_session_state():
        """Initialize all required session state variables"""
        if 'training_started' not in st.session_state:
            st.session_state.training_started = False
        if 'models_trained' not in st.session_state:
            st.session_state.models_trained = False
        if 'training_progress' not in st.session_state:
            st.session_state.training_progress = {}  # Per-coin progress
        if 'overall_progress' not in st.session_state:
            st.session_state.overall_progress = 0    # Total models completed
        if 'total_models' not in st.session_state:
            st.session_state.total_models = min(4, selected_data.shape[1]) * 4  # 4 models per coin
        if 'last_update' not in st.session_state:
            st.session_state.last_update = time.time()
        if 'training_error' not in st.session_state:
            st.session_state.training_error = None

    # Initialize session state
    initialize_session_state()
    
    # Reset state for new training
    with threading.Lock():
        st.session_state.training_started = True
        st.session_state.models_trained = False
        st.session_state.training_progress = {}  # Reset per-coin progress
        st.session_state.overall_progress = 0    # Reset overall counter
        st.session_state.training_error = None
        st.session_state.last_update = time.time()
    
    logging.info("Starting model training in background")
    
    def training_task():
        try:
            logging.info(f"Training models for {min(4, selected_data.shape[1])} coins")
            
            # Train models for each coin
            for coin_idx in range(min(4, selected_data.shape[1])):
                train_models_for_coin(selected_data, coin_idx)
                
                # Update overall progress
                with threading.Lock():
                    st.session_state.overall_progress += 4  # 4 models per coin
                    st.session_state.last_update = time.time()
            
            # Mark training as complete
            with threading.Lock():
                st.session_state.models_trained = True
                st.session_state.training_started = False
                st.session_state.last_update = time.time()
                
            logging.info("All models trained successfully")
            
        except Exception as e:
            error_msg = f"Training failed: {str(e)}"
            logging.error(error_msg, exc_info=True)
            with threading.Lock():
                st.session_state.training_error = error_msg
                st.session_state.training_started = False
            return

    # Start the background thread
    st.session_state.training_thread = threading.Thread(
        target=training_task,
        daemon=True
    )
    st.session_state.training_thread.start()
    
    logging.info("Background training thread started")

def check_training_status():
    """Check and display training progress"""
    if not isinstance(st.session_state.training_progress, dict):
        st.session_state.training_progress = {}
    
    st.subheader("Training Progress")
    
    # Show overall progress
    if hasattr(st.session_state, 'overall_progress') and hasattr(st.session_state, 'total_models'):
        overall_progress = st.session_state.overall_progress
        total_models = st.session_state.total_models
        st.progress(overall_progress / total_models)
        st.write(f"Overall progress: {overall_progress}/{total_models} models completed")
    
    # Show per-coin progress
    all_completed = True
    any_failed = False
    
    for coin_name, progress in st.session_state.training_progress.items():
        col1, col2 = st.columns([1, 4])
        with col1:
            if progress['status'] == 'Completed':
                st.success("✓")
            elif progress['status'].startswith('Failed'):
                st.error("✗")
                any_failed = True
            else:
                st.info("⌛")
                all_completed = False
        
        with col2:
            st.write(f"**{coin_name}**")
            if progress['current_model']:
                st.write(f"Current: {progress['current_model']}")
            if progress['message']:
                st.write(progress['message'])
            if progress['status'] not in ['Completed', 'Failed']:
                st.progress(progress['progress'] / 100)
    
    if all_completed and not any_failed and overall_progress >= total_models:
        st.session_state.models_trained = True
        st.balloons()
        return True
    
    # Auto-refresh every 5 seconds if training is still in progress
    if not all_completed or overall_progress < total_models:
        time.sleep(5)
        st.rerun()
    
    return False

# Fetch cryptocurrency data
def get_crypto_data(ticker, start_date, end_date):
    try:
        crypto = yf.Ticker(ticker)
        data = crypto.history(start=start_date, end=end_date)
        return data
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
        return None

# Define ticker symbols and date range
ticker_symbols = ['BTC-GBP', 'ETH-GBP', 'USDT-GBP', 'BNB-GBP', 'SOL-GBP', 'XRP-GBP', 
                 'USDC-GBP', 'ADA-GBP', 'DOGE-GBP', 'XMR-GBP', 'TRX-GBP', 'DOT-GBP', 
                 'LINK-GBP', 'MATIC-GBP', 'DAI-GBP', 'HBAR-GBP', 'ICP-GBP', 'LTC-GBP', 
                 'BCH-GBP', 'ATOM-GBP', 'ETC-GBP', 'XLM-GBP', 'MKR-GBP', 'TUSD-GBP', 
                 'HEX-GBP', 'XCH-GBP', 'FTM-GBP', 'AXS-GBP', 'NEO-GBP', 'SAND-GBP']

end_date = datetime.now()
start_date = end_date - timedelta(days=4*365)  # four years ago

# Try to load existing data or fetch fresh data
data_file = "Cleaned_combined_crypto_data.csv"
if os.path.exists(data_file):
    combined_data = pd.read_csv(data_file, index_col='Date')
else:
    combined_data = pd.DataFrame()
    for ticker in ticker_symbols:
        data = get_crypto_data(ticker, start_date, end_date)
        if data is not None:
            data['Crypto'] = ticker
            combined_data = pd.concat([combined_data, data], axis=0)
    
    if not combined_data.empty:
        combined_data.drop(['Dividends', 'Stock Splits'], axis=1, inplace=True)
        combined_data.to_csv(data_file)

# Generate selected coins through PCA and clustering
def generate_selected_data(data):
    pivoted_data = data.pivot(columns='Crypto', values='Close')
    pivoted_data_filled = pivoted_data.fillna(0)
    
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(pivoted_data_filled)
    
    pca = PCA(n_components=10)
    pca_result = pca.fit_transform(scaled_data)
    
    loadings = pd.DataFrame(pca.components_.T, 
                           columns=[f'PC{i}' for i in range(1, 11)], 
                           index=pivoted_data.columns)
    
    kmeans = KMeans(n_clusters=4, random_state=0)
    cluster_labels = kmeans.fit_predict(loadings)
    loadings['Cluster'] = cluster_labels
    
    representative_coins = pd.DataFrame()
    for i in range(4):
        cluster = loadings[loadings['Cluster'] == i]
        center = kmeans.cluster_centers_[i]
        cluster['distance_to_center'] = cluster.apply(lambda x: np.linalg.norm(x[:-1] - center), axis=1)
        representative_coins = pd.concat([representative_coins, cluster.loc[[cluster['distance_to_center'].idxmin()]]])
    
    selected_data = pivoted_data[representative_coins.index]
    return selected_data

selected_data_file = 'Selected_coins.csv'
if os.path.exists(selected_data_file):
    selected_data = pd.read_csv(selected_data_file, index_col='Date')
else:
    if not combined_data.empty:
        selected_data = generate_selected_data(combined_data)
        selected_data.to_csv(selected_data_file)
    else:
        selected_data = pd.DataFrame()

# Check if models need to be trained on first run
if not os.path.exists("trained_models") and not selected_data.empty:
    st.info("First-time setup: Training initial models...")
    train_all_models_background(selected_data)

# UI Functions
def home_section():
    st.title("SOLiGence")
    st.write(
        "Solent Intelligence (SOLiGence) is a leading financial multinational organisation that deals"
        " with stock and shares, saving and investments.")

    st.header('Welcome to SOLiGence')
    st.write("Your Intelligent Coin Trading Platform")
    st.write("Empower your cryptocurrency trading decisions with AI-driven insights and real-time data.")

    if st.button("Get Started"):
        st.write("Let's explore the world of cryptocurrency trading together!")
        st.image('https://img.freepik.com/free-vector/gradient-stock-market-concept_23-2149166910.jpg', 
                use_container_width=True)
        
        st.write("News and Updates:")
        st.info("Stay tuned for the latest updates and trends in the cryptocurrency market!")
        
        st.write("What Our Users Say:")
        st.write("The SOLiGence app transformed how I approach cryptocurrency trading. Highly recommended!")
        
        st.write("Contact Us:")
        st.write("For inquiries, email us at info@soligence.com")
        
        st.write("Connect with Us:")
        st.markdown("[Twitter](https://twitter.com) [LinkedIn](https://linkedin.com)")

def about_us():
    st.title("About Solent Intelligence Ltd.")
    st.write(
        "The scale of this organization's operation is impressive, with millions of subscribers and over 150 billion "
        "pounds worth of investments. This emphasizes the substantial influence that data-driven decisions can have "
        "on managing such a significant amount of assets. The app's focus on implementing an Intelligent Coin Trading "
        "(IST) platform, specifically tailored for crypto coin predictions, resonates deeply with me.")
    
    st.write("The app's ability to recommend trading opportunities by analyzing "
        "AI-generated predictions showcases the tangible applications of data science in the financial world. "
        "Considering a more neutral perspective, while the concept of the app is exciting, there are potential "
        "challenges that need to be acknowledged.")

def dataset_section():
    st.header("Crypto Dataset of 30 Coins")
    
    if combined_data.empty:
        st.error("No data available. Please check your data source.")
        return
    
    st.sidebar.subheader("Dataset Options")
    sort_column = st.sidebar.multiselect("Sort by:", combined_data.columns)
    ascending = st.sidebar.checkbox("Ascending", True)
    
    sorted_data = combined_data.sort_values(by=sort_column, ascending=ascending)
    selected_crypto = st.sidebar.selectbox("Filter by cryptocurrency:", ['All'] + list(combined_data['Crypto'].unique()))
    
    if selected_crypto != 'All':
        sorted_data = sorted_data[sorted_data['Crypto'] == selected_crypto]
    
    page_size = st.sidebar.number_input("Items per page:", min_value=1, value=10)
    page_number = st.sidebar.number_input("Page number:", min_value=1, value=1)
    
    start_idx = (page_number - 1) * page_size
    end_idx = start_idx + page_size
    paginated_data = sorted_data.iloc[start_idx:end_idx]
    
    st.subheader("Dataset Overview")
    st.dataframe(paginated_data)
    
    if st.checkbox("Show as Table"):
        st.subheader("Dataset Table View")
        st.table(paginated_data)

def plot_average_price_trend(data, selected_coin, interval):
    selected_coin_data = data[data['Crypto'] == selected_coin]
    
    if selected_coin_data.empty:
        st.error("No data available for the selected cryptocurrency.")
        return
    
    selected_coin_data.index = pd.to_datetime(selected_coin_data.index)
    
    if interval == 'Daily':
        resampled_data = selected_coin_data['Close']
        interval_label = 'Daily'
    elif interval == 'Weekly':
        resampled_data = selected_coin_data['Close'].resample('W').mean()
        interval_label = 'Weekly'
    elif interval == 'Monthly':
        resampled_data = selected_coin_data['Close'].resample('M').mean()
        interval_label = 'Monthly'
    else:
        st.error("Invalid interval.")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(resampled_data.index.date, resampled_data, color='blue', marker='o', linestyle='-')
    ax.set_xlabel('Date', fontsize=14)
    ax.set_ylabel('Average Price', fontsize=14)
    ax.set_title(f'Average {interval_label} Price Trend for {selected_coin}', fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    st.pyplot(fig)

def plot_crypto_volatility():
    st.title("Cryptocurrency Volume Volatility")
    
    if combined_data.empty:
        st.error("No data available.")
        return
    
    available_coins = combined_data['Crypto'].unique()
    selected_coins = st.multiselect("Select cryptocurrencies:", options=available_coins, default=available_coins[:2])
    
    if not selected_coins:
        return
    
    filtered_data = combined_data[combined_data['Crypto'].isin(selected_coins)]
    window_size = 7
    fig = go.Figure(layout_title_text="Volume Volatility Over Time",
                   layout_xaxis_title="Date",
                   layout_yaxis_title="Volume Volatility (Standard Deviation)")

    for coin in selected_coins:
        coin_data = filtered_data[filtered_data['Crypto'] == coin]['Volume'].rolling(window=window_size).std()
        fig.add_trace(go.Scatter(x=coin_data.index, y=coin_data, mode='lines', name=coin))

    st.plotly_chart(fig)

def plot_distribution_and_trend(selected_coin):
    selected_data = combined_data[combined_data['Crypto'] == selected_coin]
    
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=selected_data['Close'], nbinsx=30,
                             histnorm='probability density', 
                             marker_color='skyblue',
                             opacity=0.7, name='Histogram'))
    
    kde = gaussian_kde(selected_data['Close'], bw_method=0.2)
    kde_x = np.linspace(selected_data['Close'].min(), selected_data['Close'].max(), 100)
    kde_y = kde.evaluate(kde_x)
    
    fig.add_trace(go.Scatter(x=kde_x, y=kde_y, mode='lines', 
                           name='Trend Line', line=dict(color='orange')))
    
    fig.update_layout(title=f'Distribution of Close Prices for {selected_coin}',
                    xaxis_title='Close Price', 
                    yaxis_title='Probability Density')
    st.plotly_chart(fig)

def plot_daily_price_changes(selected_coin):
    selected_data = combined_data[combined_data['Crypto'] == selected_coin]
    selected_data['Price Change'] = selected_data['Close'].diff()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=selected_data.index, y=selected_data['Price Change'], 
                           mode='lines', name='Price Change', line=dict(color='blue')))
    
    fig.update_layout(title=f'Daily Price Changes for {selected_coin}',
                    xaxis_title='Date', 
                    yaxis_title='Price Change',
                    width=800, height=500)
    st.plotly_chart(fig)

def plot_boxplot(data, selected_coin):
    coin_data = data[data['Crypto'] == selected_coin]
    
    fig = go.Figure()
    for metric in ['Low', 'Close', 'Open', 'High']:
        fig.add_trace(go.Box(y=coin_data[metric], name=metric))
    
    fig.update_layout(title=f"Boxplot for {selected_coin}",
                    xaxis_title="Metrics",
                    yaxis_title="Price",
                    showlegend=True)
    st.plotly_chart(fig)

def visualize_crypto_data():
    unique_coins = combined_data['Crypto'].unique()
    selected_coin = st.selectbox("Select cryptocurrency:", unique_coins)
    
    selected_coin_data = combined_data[combined_data['Crypto'] == selected_coin]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=selected_coin_data.index, 
                           y=selected_coin_data['Close'], 
                           mode='lines', 
                           name='Close', 
                           line=dict(color='orange'), 
                           yaxis='y2'))
    
    fig.add_trace(go.Scatter(x=selected_coin_data.index, 
                           y=selected_coin_data['Volume'], 
                           fill='tozeroy', 
                           name='Volume', 
                           line=dict(color='gray', width=0), 
                           fillcolor='rgba(128, 128, 128, 0.3)'))
    
    fig.update_layout(title=f'Price and Volume for {selected_coin}',
                    xaxis=dict(title='Date'),
                    yaxis=dict(title='Volume', side='left', showgrid=False),
                    yaxis2=dict(title='Price', side='right', overlaying='y', showgrid=False),
                    legend=dict(x=0, y=1))
    st.plotly_chart(fig)

def analyze_coin_correlation():
    if combined_data.empty:
        st.error("No data available.")
        return
    
    pivoted_data = combined_data.pivot(columns='Crypto', values='Close')
    coins_list = pivoted_data.columns.tolist()
    
    coin_selected = st.selectbox("Select coin:", coins_list)
    selected_coin_prices = pivoted_data[coin_selected]
    correlations = pivoted_data.corrwith(selected_coin_prices)
    sorted_correlations = correlations.sort_values(ascending=False)
    sorted_correlations = sorted_correlations.drop(coin_selected)
    
    st.write(f"Top positively correlated with {coin_selected}:")
    st.write(sorted_correlations.head(4))
    
    st.write(f"Top negatively correlated with {coin_selected}:")
    st.write(sorted_correlations.tail(4))

def plot_moving_average():
    available_coins = combined_data['Crypto'].unique()
    coin_selected = st.selectbox("Select cryptocurrency:", available_coins)
    window_size = st.radio("Window size:", ['Short (30-day MA)', 'Medium (60-day MA)', 'Long (90-day MA)'])
    
    window_mapping = {'Short (30-day MA)': 30, 'Medium (60-day MA)': 60, 'Long (90-day MA)': 90}
    window = window_mapping[window_size]
    
    selected_data = combined_data[combined_data['Crypto'] == coin_selected].copy()
    selected_data['MA'] = selected_data['Close'].rolling(window=window).mean()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=selected_data.index, y=selected_data['Close'], name=f'{coin_selected} Price'))
    fig.add_trace(go.Scatter(x=selected_data.index, y=selected_data['MA'], name=f'{window}-day MA'))
    
    fig.update_layout(title=f'Moving Average for {coin_selected}',
                    xaxis_title='Date',
                    yaxis_title='Price')
    st.plotly_chart(fig)

def plot_crypto_metrics(coin, metrics, start_date, end_date):
    selected_data = combined_data[combined_data['Crypto'] == coin]
    selected_data = selected_data.loc[start_date:end_date]
    
    fig = go.Figure()
    for metric in metrics:
        fig.add_trace(go.Scatter(x=selected_data.index, y=selected_data[metric], name=f"{coin} {metric}"))
    
    fig.update_layout(title=f'Prices for {coin} ({start_date} to {end_date})',
                    xaxis_title='Date',
                    yaxis_title='Price')
    st.plotly_chart(fig)

def plot_crypto_coins(coins, metric, start_date, end_date):
    fig = go.Figure()
    for coin in coins:
        coin_data = combined_data[combined_data['Crypto'] == coin]
        coin_data = coin_data.loc[start_date:end_date]
        fig.add_trace(go.Scatter(x=coin_data.index, y=coin_data[metric], name=f"{coin} {metric}"))
    
    fig.update_layout(title=f'{metric} Prices ({start_date} to {end_date})',
                    xaxis_title='Date',
                    yaxis_title=metric)
    st.plotly_chart(fig)

def plot_candlestick_chart(coin='BTC-GBP', period='D'):
    coin_data = combined_data[combined_data['Crypto'] == coin].copy()
    coin_data.index = pd.to_datetime(coin_data.index)
    
    if period.upper() == 'W':
        resampled_data = coin_data.resample('W').agg({'Open': 'first', 
                                                     'High': 'max', 
                                                     'Low': 'min', 
                                                     'Close': 'last', 
                                                     'Volume': 'sum'})
    elif period.upper() == 'M':
        resampled_data = coin_data.resample('M').agg({'Open': 'first', 
                                                     'High': 'max', 
                                                     'Low': 'min', 
                                                     'Close': 'last', 
                                                     'Volume': 'sum'})
    else:
        resampled_data = coin_data

    fig = go.Figure(data=[go.Candlestick(x=resampled_data.index,
                                       open=resampled_data['Open'],
                                       high=resampled_data['High'],
                                       low=resampled_data['Low'],
                                       close=resampled_data['Close'],
                                       name='Candlestick'),
                        go.Bar(x=resampled_data.index,
                               y=resampled_data['Volume'],
                               name='Volume',
                               marker_color='rgba(0, 0, 0, 0.5)')])

    fig.update_layout(title=f'{coin} {period.upper()} Chart',
                    xaxis_title='Date',
                    yaxis_title='Price')
    st.plotly_chart(fig)

def visualize_market_state():
    available_coins = combined_data['Crypto'].unique()
    coin_selected = st.selectbox("Select cryptocurrency:", available_coins)
    
    selected_data = combined_data[combined_data['Crypto'] == coin_selected].copy()
    selected_data['Close'] = pd.to_numeric(selected_data['Close'], errors='coerce')
    selected_data.dropna(subset=['Close'], inplace=True)
    
    price_changes = selected_data['Close'].pct_change()
    price_changes.replace([np.inf, -np.inf], np.nan, inplace=True)
    price_changes.dropna(inplace=True)
    
    market_state = np.sign(price_changes).astype(float)
    
    st.write("Explanation:")
    st.write("Down: -1.0 suggests market decrease")
    st.write("Up: 1.0 suggests market increase")

    fig, ax = plt.subplots(figsize=(10, 6))
    selected_data['Close'].plot(ax=ax, label='Close Price')
    ax.scatter(selected_data.index, selected_data['Close'], c='g', label='Up Market', marker='^', alpha=0.5)
    ax.scatter(selected_data.index, selected_data['Close'], c='r', label='Down Market', marker='v', alpha=0.5)
    ax.set_title(f'Market State for {coin_selected}')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend()
    st.pyplot(fig)

def predict_highs_lows():
    available_coins = combined_data['Crypto'].unique()
    coin_selected = st.selectbox("Select cryptocurrency:", available_coins)
    
    selected_data = combined_data[combined_data['Crypto'] == coin_selected].copy()
    high_threshold = np.percentile(selected_data['High'], 90)
    low_threshold = np.percentile(selected_data['Low'], 10)
    
    predicted_highs = selected_data['Close'] > high_threshold
    predicted_lows = selected_data['Close'] < low_threshold
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=selected_data.index, y=selected_data['Close'], name='Close Price'))
    fig.add_trace(go.Scatter(x=selected_data.index[predicted_highs], 
                           y=selected_data['Close'][predicted_highs],
                           mode='markers', name='Predicted Highs',
                           marker=dict(color='red', size=8)))
    fig.add_trace(go.Scatter(x=selected_data.index[predicted_lows], 
                           y=selected_data['Close'][predicted_lows],
                           mode='markers', name='Predicted Lows',
                           marker=dict(color='green', size=8)))
    
    fig.update_layout(title=f'Predicted Highs/Lows for {coin_selected}',
                    xaxis_title='Date',
                    yaxis_title='Price')
    st.plotly_chart(fig)

def display_selected_coins():
    st.title("Selected Coins Analysis")
    
    if selected_data.empty:
        st.error("No selected coins data available.")
        return
    
    st.dataframe(selected_data)
    
    fig = make_subplots(rows=1, cols=4)
    for i, column in enumerate(selected_data.columns[:4], start=1):
        fig.add_trace(go.Box(y=selected_data[column], name=column), row=1, col=i)
    
    fig.update_layout(title='Boxplot of Selected Coins',
                    showlegend=False,
                    width=1000, height=500)
    st.plotly_chart(fig)

def plot_coin_scatter():
    if selected_data.empty:
        st.error("No selected coins data available.")
        return
    
    coin_combinations = list(itertools.combinations(selected_data.columns, 2))
    fig = make_subplots(rows=len(coin_combinations)//3 + 1, cols=3,
                       subplot_titles=[f"{coin1} vs {coin2}" for coin1, coin2 in coin_combinations])
    
    for i, (coin1, coin2) in enumerate(coin_combinations, start=1):
        row = (i - 1) // 3 + 1
        col = (i - 1) % 3 + 1
        fig.add_trace(go.Scatter(x=selected_data[coin1], y=selected_data[coin2], 
                       mode='markers', name=f"{coin1} vs {coin2}"), row=row, col=col)
    
    fig.update_layout(height=800, width=1000, showlegend=False)
    st.plotly_chart(fig)

def evaluate_models_selected_coin(data, coin_index, chosen_model='all'):
    """
    Evaluate machine learning models for a specific cryptocurrency with version compatibility checks
    and automatic recovery if models fail to load.
    """
    try:
        # Validate input data
        if data.empty:
            st.error("No data available for evaluation.")
            return
            
        coin_name = data.columns[coin_index]
        model_dir = f"trained_models/Model_SELECTED_COIN_{coin_index+1}"
        
        # Display version requirements if they exist
        req_file = f"{model_dir}/requirements.txt"
        if os.path.exists(req_file):
            with open(req_file) as f:
                st.info(f"Model version requirements:\n```\n{f.read()}\n```")
        
        # Check if models exist, if not train them
        if not os.path.exists(model_dir):
            st.warning(f"No trained models found for {coin_name}. Training models now...")
            train_models_for_coin(data, coin_index)
            st.success("Models trained successfully!")
        
        # Prepare data with lag features
        data_prep = data.copy()
        for lag in range(1, 4):
            data_prep[f'{coin_name}_lag_{lag}'] = data_prep[coin_name].shift(lag)
        data_prep.dropna(inplace=True)
        
        features = [f'{coin_name}_lag_{lag}' for lag in range(1, 4)]
        X = data_prep[features]
        y = data_prep[coin_name]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Model loading with error recovery
        models = {}
        retrained = False
        try:
            models = {
                'GRADIENT BOOSTING': joblib.load(f"{model_dir}/gradient_boosting_model.pkl"),
                'SVR': joblib.load(f"{model_dir}/svr_model.pkl"),
                'XGBOOST': joblib.load(f"{model_dir}/xgboost_model.pkl"),
                'LSTM': tf.keras.models.load_model(f"{model_dir}/lstm_model.keras")
            }
        except Exception as e:
            st.warning(f"Model loading failed due to version mismatch. Retraining models...")
            with st.spinner("Retraining models..."):
                train_models_for_coin(data, coin_index)
                retrained = True
                
                # Second attempt after retraining
                try:
                    models = {
                        'GRADIENT BOOSTING': joblib.load(f"{model_dir}/gradient_boosting_model.pkl"),
                        'SVR': joblib.load(f"{model_dir}/svr_model.pkl"),
                        'XGBOOST': joblib.load(f"{model_dir}/xgboost_model.pkl"),
                        'LSTM': tf.keras.models.load_model(f"{model_dir}/lstm_model.keras")
                    }
                except Exception as e:
                    st.error(f"Model loading failed again: {str(e)}")
                    return

        # Determine which models to evaluate
        if chosen_model.lower() == 'all':
            models_to_evaluate = models.keys()
        else:
            chosen_model = chosen_model.upper()
            if chosen_model in models:
                models_to_evaluate = [chosen_model]
            else:
                st.error(f"Model '{chosen_model}' not found in trained models")
                return

        # Evaluation metrics storage
        eval_metrics = {}
        predictions_data = []

        for model_name in models_to_evaluate:
            if models[model_name] is None:
                st.warning(f"Skipping {model_name} - model not available")
                continue

            try:
                # Make predictions
                if model_name == 'LSTM':
                    X_test_array = X_test.to_numpy().reshape(X_test.shape[0], X_test.shape[1], 1)
                    predictions = models[model_name].predict(X_test_array).flatten()
                else:
                    predictions = models[model_name].predict(X_test)

                # Calculate metrics
                metrics = {
                    'MAE': mean_absolute_error(y_test, predictions),
                    'MSE': mean_squared_error(y_test, predictions),
                    'RMSE': np.sqrt(mean_squared_error(y_test, predictions)),
                    'MAPE': np.mean(np.abs((y_test - predictions) / y_test)) * 100,
                    'R2': r2_score(y_test, predictions)
                }
                eval_metrics[model_name] = metrics

                # Store prediction data for visualization
                predictions_data.append({
                    'Model': model_name,
                    'Actual': y_test,
                    'Predicted': predictions
                })

            except Exception as e:
                st.error(f"Error evaluating {model_name}: {str(e)}")
                continue

        # Display results
        if not eval_metrics:
            st.error("No models were successfully evaluated")
            return

        st.subheader(f"Evaluation Results for {coin_name}")
        
        # Metrics table
        metrics_df = pd.DataFrame.from_dict(eval_metrics, orient='index')
        st.dataframe(metrics_df.style.format({
            'MAE': '{:.4f}',
            'MSE': '{:.4f}',
            'RMSE': '{:.4f}',
            'MAPE': '{:.2f}%',
            'R2': '{:.4f}'
        }))

        # Metrics visualization
        st.subheader("Model Comparison")
        fig = go.Figure()
        for metric in ['MAE', 'RMSE', 'R2']:
            fig.add_trace(go.Bar(
                x=metrics_df.index,
                y=metrics_df[metric],
                name=metric,
                text=metrics_df[metric].round(4),
                textposition='auto'
            ))
        fig.update_layout(
            barmode='group',
            title='Model Performance Comparison',
            xaxis_title='Model',
            yaxis_title='Metric Value'
        )
        st.plotly_chart(fig)

        # Actual vs Predicted visualization
        st.subheader("Actual vs Predicted Values")
        fig2 = go.Figure()
        for pred_data in predictions_data:
            fig2.add_trace(go.Scatter(
                x=y_test,
                y=pred_data['Predicted'],
                mode='markers',
                name=pred_data['Model'],
                marker=dict(size=8, opacity=0.6)
            ))
        # Add perfect prediction line
        fig2.add_trace(go.Scatter(
            x=[y_test.min(), y_test.max()],
            y=[y_test.min(), y_test.max()],
            mode='lines',
            name='Perfect Prediction',
            line=dict(color='black', dash='dash')
        ))
        fig2.update_layout(
            title='Actual vs Predicted Values',
            xaxis_title='Actual Price',
            yaxis_title='Predicted Price',
            showlegend=True
        )
        st.plotly_chart(fig2)

        # Show retrained notice if applicable
        if retrained:
            st.info("Note: Models were retrained with current environment settings")

    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        st.error("Please check your data and model files")

def plot_actual_forecast_with_confidence(actual, predictions, periods, upper_bound, lower_bound):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=periods, y=actual, mode='lines', name='Actual', line=dict(color='green')))
    fig.add_trace(go.Scatter(x=periods, y=predictions, mode='lines', name='Forecast', line=dict(color='red')))
    fig.add_trace(go.Scatter(x=periods, y=upper_bound, mode='lines', name='Upper Bound', line=dict(color='blue', width=0)))
    fig.add_trace(go.Scatter(x=periods, y=lower_bound, mode='lines', name='Lower Bound', fill='tonexty', line=dict(color='blue')))
    
    fig.update_layout(title="Actual vs Forecast with Confidence",
                    xaxis_title='Date',
                    yaxis_title='Price')
    st.plotly_chart(fig)

def evaluate_and_plot_model(coin_index, model_choice, frequency, num_periods):
    if selected_data.empty:
        st.error("No selected coins data available.")
        return
    
    coin_name = selected_data.columns[coin_index]
    
    for lag in range(1, 4):
        selected_data[f'{coin_name}_lag_{lag}'] = selected_data[coin_name].shift(lag)
    
    selected_data.dropna(inplace=True)
    features = [f'{coin_name}_lag_{lag}' for lag in range(1, 4)]
    X = selected_data[features]
    y = selected_data[coin_name]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model_filename = f"trained_models/Model_SELECTED_COIN_{coin_index+1}/"
    
    # Create a mapping between display names and actual filenames
    model_mapping = {
        'Gradient Boosting': 'gradient_boosting_model.pkl',
        'SVR': 'svr_model.pkl',
        'XGBOOST': 'xgboost_model.pkl',
        'LSTM': 'lstm_model.keras'
    }
    
    if model_choice in model_mapping:
        model_filename += model_mapping[model_choice]
        
        if model_choice == 'LSTM':
            if os.path.exists(model_filename):
                model = load_model(model_filename)
                X_array = X.to_numpy().reshape(X.shape[0], X.shape[1], 1)
                predictions = model.predict(X_array[-num_periods:]).flatten()
            else:
                st.error("LSTM model not found.")
                return
        else:
            if os.path.exists(model_filename):
                model = joblib.load(model_filename)
                predictions = model.predict(X[-num_periods:])
            else:
                st.error(f"{model_choice} model not found at {model_filename}")
                return
    else:
        st.error("Invalid model selection.")
        return
    
    # .
    last_date = selected_data.index[-1]
    if frequency == 'daily':
        periods = pd.date_range(start=last_date, periods=num_periods, freq='D')
    elif frequency == 'weekly':
        periods = pd.date_range(start=last_date, periods=num_periods, freq='W')
    elif frequency == 'monthly':
        periods = pd.date_range(start=last_date, periods=num_periods, freq='M')
    elif frequency == 'quarterly':
        periods = pd.date_range(start=last_date, periods=num_periods, freq='Q')
    else:
        st.error("Invalid frequency.")
        return
    
    mse = mean_squared_error(y_test[-num_periods:], predictions)
    upper_bound = predictions + 1.96 * np.sqrt(mse)
    lower_bound = predictions - 1.96 * np.sqrt(mse)
    
    predictions_df = pd.DataFrame({'Date': periods, 'Predictions': predictions})
    st.dataframe(predictions_df)
    
    plot_actual_forecast_with_confidence(y_test[-num_periods:], predictions, periods, upper_bound, lower_bound)



from datetime import datetime, timedelta
import pandas as pd
from ta.trend import SMAIndicator
import os
import joblib
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta
from tensorflow.keras.models import load_model
from ta.trend import SMAIndicator
def apply_ma_trading_strategy(chosen_coin, combined_data):
    """Apply moving average trading strategy with error handling"""
    try:
        # Filter data for selected coin
        selected_data = combined_data[combined_data['Crypto'] == chosen_coin].copy()
        selected_data.dropna(subset=['Close'], inplace=True)
        
        # Handle datetime index safely
        if not isinstance(selected_data.index, pd.DatetimeIndex):
            selected_data.index = pd.to_datetime(selected_data.index)
        if hasattr(selected_data.index, 'tz') and selected_data.index.tz is not None:
            selected_data.index = selected_data.index.tz_localize(None)
        
        # Calculate moving averages
        ma_7, ma_14 = 7, 14
        selected_data[f'MA_{ma_7}'] = SMAIndicator(
            close=selected_data['Close'], 
            window=ma_7
        ).sma_indicator()
        selected_data[f'MA_{ma_14}'] = SMAIndicator(
            close=selected_data['Close'], 
            window=ma_14
        ).sma_indicator()
        
        # Generate signals
        selected_data['Buy_Signal'] = np.where(
            selected_data[f'MA_{ma_7}'] > selected_data[f'MA_{ma_14}'].shift(1), 1, 0
        )
        selected_data['Sell_Signal'] = np.where(
            selected_data[f'MA_{ma_7}'] < selected_data[f'MA_{ma_14}'].shift(1), -1, 0
        )
        
        return selected_data
        
    except Exception as e:
        st.error(f"Error generating trading strategy: {str(e)}")
        logging.error(f"Error in apply_ma_trading_strategy: {str(e)}", exc_info=True)
        return None

def forecast_price(selected_data, chosen_coin, num_days):
    """Forecast future price using moving averages"""
    try:
        future_date = datetime.now() + timedelta(days=num_days)
        future_date = future_date.replace(tzinfo=None)
        
        last_date = selected_data.index[-1].to_pydatetime().replace(tzinfo=None)
        
        if future_date > last_date:
            new_index = pd.date_range(start=selected_data.index[0], periods=len(selected_data) + num_days, freq='D')
            extended_data = selected_data.reindex(new_index, method='ffill')
            
            ma_7 = 7
            ma_14 = 14
            
            extended_data[f'MA_{ma_7}'] = SMAIndicator(close=extended_data['Close'], window=ma_7).sma_indicator()
            extended_data[f'MA_{ma_14}'] = SMAIndicator(close=extended_data['Close'], window=ma_14).sma_indicator()
            
            future_price = (extended_data[f'MA_{ma_7}'].iloc[-1] + extended_data[f'MA_{ma_14}'].iloc[-1]) / 2
            return future_price, future_date
        
        return None, None
    except Exception as e:
        st.error(f"Error in price forecasting: {str(e)}")
        logging.error(f"Error in forecast_price: {str(e)}", exc_info=True)
        return None, None

def determine_best_time_to_trade_future(chosen_coin, num_days, combined_data):
    """Generate trading recommendation based on moving averages"""
    try:
        selected_data = apply_ma_trading_strategy(chosen_coin, combined_data)
        if selected_data is None:
            return None
            
        future_price, future_date = forecast_price(selected_data, chosen_coin, num_days)
        
        if future_price is not None:
            current_price = selected_data['Close'].iloc[-1]
            price_change = future_price - current_price
            percentage_change = (price_change / current_price) * 100
            action = "Buy" if future_price > current_price else "Sell"
            confidence = "Strong" if abs(percentage_change) > 5 else "Moderate" if abs(percentage_change) > 2 else "Weak"
            
            # Display results
            st.markdown("""
            <style>
            .result-box {
                padding: 20px;
                border-radius: 10px;
                margin-bottom: 20px;
                background-color: #f0f2f6;
                border-left: 5px solid #4e8cff;
            }
            .metric-label {
                font-size: 14px;
                color: #555;
                font-weight: bold;
            }
            .metric-value {
                font-size: 24px;
                font-weight: bold;
                margin-bottom: 5px;
            }
            .price-up {
                color: #10b981;
            }
            .price-down {
                color: #ef4444;
            }
            </style>
            """, unsafe_allow_html=True)
            
            result_html = f"""
            <div class="result-box">
                <div style="display: flex; justify-content: space-between; flex-wrap: wrap;">
                    <div style="min-width: 150px; margin-right: 10px; margin-bottom: 15px;">
                        <div class="metric-label">Current Price</div>
                        <div class="metric-value">${current_price:.4f}</div>
                    </div>
                    <div style="min-width: 150px; margin-right: 10px; margin-bottom: 15px;">
                        <div class="metric-label">Predicted Price</div>
                        <div class="metric-value {'price-up' if price_change > 0 else 'price-down'}">${future_price:.4f}</div>
                        <div>{'▲' if price_change > 0 else '▼'} {abs(percentage_change):.2f}%</div>
                    </div>
                    <div style="min-width: 150px; margin-bottom: 15px;">
                        <div class="metric-label">Forecast Date</div>
                        <div class="metric-value">{future_date.strftime('%Y-%m-%d')}</div>
                        <div>({num_days} days ahead)</div>
                    </div>
                </div>
                <div style="margin-top: 15px; padding-top: 15px; border-top: 1px solid #ddd;">
                    <div style="font-size: 18px; font-weight: bold; margin-bottom: 5px;">
                        Recommendation: <span style="color: {'#10b981' if action == 'Buy' else '#ef4444'}">{action}</span> with {confidence} confidence
                    </div>
                    <div style="font-style: italic; color: #666;">
                        Based on moving average analysis of {chosen_coin}'s historical data
                    </div>
                </div>
            </div>
            """
            
            st.markdown(result_html, unsafe_allow_html=True)
            st.caption("Note: This forecast is an estimate and market conditions can change unexpectedly.")
            
            # Plot the moving averages
            plot_ma_strategy(selected_data, chosen_coin)
            
        else:
            st.error("Unable to generate forecast. Please check your data inputs.")
        
        return selected_data
        
    except Exception as e:
        st.error(f"Error in trading recommendation: {str(e)}")
        logging.error(f"Error in determine_best_time_to_trade_future: {str(e)}", exc_info=True)
        return None

# def plot_ma_strategy(selected_data, chosen_coin):
#     """Plot the moving average strategy"""
#     try:
#         fig = go.Figure()
        
#         # Add traces with proper formatting
#         traces = [
#             ('Close Price', 'blue', None, selected_data['Close']),
#             ('7-day MA', 'green', None, selected_data['MA_7']),
#             ('14-day MA', 'red', None, selected_data['MA_14']),
#             ('Buy Signal', 'green', 'triangle-up', 
#              selected_data.loc[selected_data['Buy_Signal'] == 1, 'Close']),
#             ('Sell Signal', 'red', 'triangle-down',
#              selected_data.loc[selected_data['Sell_Signal'] == -1, 'Close'])
#         ]
        
#         for name, color, symbol, y in traces:
#             fig.add_trace(go.Scatter(
#                 x=selected_data.index,
#                 y=y,
#                 name=name,
#                 mode='lines' if symbol is None else 'markers',
#                 line=dict(color=color) if symbol is None else None,
#                 marker=dict(color=color, size=10, symbol=symbol) if symbol else None
#             ))
        
#         # Add current price line
#         current_price = selected_data['Close'].iloc[-1]
#         fig.add_trace(go.Scatter(
#             x=[selected_data.index[0], selected_data.index[-1]],
#             y=[current_price, current_price],
#             mode='lines',
#             name='Current Price',
#             line=dict(color='gray', dash='dash')
#         )
        
#         # Update layout
#         fig.update_layout(
#             title=f'Moving Average Strategy for {chosen_coin}',
#             xaxis_title='Date',
#             yaxis_title='Price',
#             hovermode='x unified',
#             showlegend=True
#         )
        
#         st.plotly_chart(fig, use_container_width=True)
        
#     except Exception as e:
#         st.error(f"Error plotting strategy: {str(e)}")
#         logging.error(f"Error in plot_ma_strategy: {str(e)}", exc_info=True)

def forecast_price_with_model(chosen_coin, num_days, model_type, selected_data):
    """Forecast using machine learning models"""
    try:
        if chosen_coin not in selected_data.columns:
            st.error(f"Selected coin '{chosen_coin}' not found in data")
            return None, None
            
        coin_index = selected_data.columns.get_loc(chosen_coin)
        model_dir = f"trained_models/Model_SELECTED_COIN_{coin_index+1}"
        
        model_mapping = {
            "SVR": "svr_model.pkl",
            "GBR": "gradient_boosting_model.pkl",
            "XGBoost": "xgboost_model.pkl",
            "LSTM": "lstm_model.keras"
        }
        
        if model_type not in model_mapping:
            st.error(f"Invalid model type: {model_type}")
            return None, None
            
        model_filename = os.path.join(model_dir, model_mapping[model_type])
        
        if not os.path.exists(model_filename):
            st.error(f"Model not found: {model_filename}")
            return None, None
        
        features = [f'{chosen_coin}_lag_{lag}' for lag in range(1, 4)]
        
        data_copy = selected_data.copy()
        for lag in range(1, 4):
            lag_col = f'{chosen_coin}_lag_{lag}'
            if lag_col not in data_copy.columns:
                data_copy[lag_col] = data_copy[chosen_coin].shift(lag)
        
        selected_data_clean = data_copy.dropna(subset=features)
        
        if len(selected_data_clean) == 0:
            st.error("Not enough historical data to generate forecast")
            return None, None
        
        X_array = selected_data_clean[features].to_numpy()
        
        with st.spinner(f"Predicting future price with {model_type} model..."):
            if model_type == "LSTM":
                try:
                    model = load_model(model_filename)
                    X_today = X_array[-1].reshape(1, len(features), 1)
                    future_price = model.predict(X_today)[0][0]
                except Exception as e:
                    st.error(f"Error with LSTM prediction: {str(e)}")
                    return None, None
            else:
                try:
                    model = joblib.load(model_filename)
                    X_today = X_array[-1].reshape(1, -1)
                    future_price = model.predict(X_today)[0]
                except Exception as e:
                    st.error(f"Error with {model_type} prediction: {str(e)}")
                    return None, None
        
        future_date = datetime.now() + timedelta(days=num_days)
        return future_price, future_date
    
    except Exception as e:
        st.error(f"Error in price forecasting: {str(e)}")
        logging.error(f"Error in forecast_price_with_model: {str(e)}", exc_info=True)
        return None, None

def create_prediction_interface(selected_data):
    """Create unified prediction interface"""
    st.markdown("## Cryptocurrency Price Prediction")
    
    with st.form(key="prediction_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            available_coins = selected_data.columns[:4].tolist() if not selected_data.empty else []
            chosen_coin = st.selectbox("Select Cryptocurrency", options=available_coins)
        
        with col2:
            model_type = st.selectbox(
                "Select Model Type",
                options=["SVR", "GBR", "XGBoost", "LSTM"]
            )
        
        with col3:
            num_days = st.slider(
                "Prediction Days Ahead",
                min_value=1,
                max_value=30,
                value=7
            )
        
        predict_button = st.form_submit_button("Predict Price")
        
        if predict_button:
            with st.spinner("Analyzing market data..."):
                future_price, future_date = forecast_price_with_model(chosen_coin, num_days, model_type, selected_data)
                
                if future_price is not None:
                    current_price = selected_data[chosen_coin].iloc[-1]
                    price_change = future_price - current_price
                    percentage_change = (price_change / current_price) * 100
                    action = "Buy" if future_price > current_price else "Sell"
                    confidence = "Strong" if abs(percentage_change) > 5 else "Moderate" if abs(percentage_change) > 2 else "Weak"
                    
                    st.markdown("""
                    <style>
                    .result-box {
                        padding: 20px;
                        border-radius: 10px;
                        margin-bottom: 20px;
                        background-color: #f0f2f6;
                        border-left: 5px solid #4e8cff;
                    }
                    .metric-label {
                        font-size: 14px;
                        color: #555;
                        font-weight: bold;
                    }
                    .metric-value {
                        font-size: 24px;
                        font-weight: bold;
                        margin-bottom: 5px;
                    }
                    .price-up {
                        color: #10b981;
                    }
                    .price-down {
                        color: #ef4444;
                    }
                    </style>
                    """, unsafe_allow_html=True)
                    
                    result_html = f"""
                    <div class="result-box">
                        <div style="display: flex; justify-content: space-between; flex-wrap: wrap;">
                            <div style="min-width: 150px; margin-right: 10px; margin-bottom: 15px;">
                                <div class="metric-label">Current Price</div>
                                <div class="metric-value">${current_price:.4f}</div>
                            </div>
                            <div style="min-width: 150px; margin-right: 10px; margin-bottom: 15px;">
                                <div class="metric-label">Predicted Price</div>
                                <div class="metric-value {'price-up' if price_change > 0 else 'price-down'}">${future_price:.4f}</div>
                                <div>{'▲' if price_change > 0 else '▼'} {abs(percentage_change):.2f}%</div>
                            </div>
                            <div style="min-width: 150px; margin-bottom: 15px;">
                                <div class="metric-label">Forecast Date</div>
                                <div class="metric-value">{future_date.strftime('%Y-%m-%d')}</div>
                                <div>({num_days} days ahead)</div>
                            </div>
                        </div>
                        <div style="margin-top: 15px; padding-top: 15px; border-top: 1px solid #ddd;">
                            <div style="font-size: 18px; font-weight: bold; margin-bottom: 5px;">
                                Recommendation: <span style="color: {'#10b981' if action == 'Buy' else '#ef4444'}">{action}</span> with {confidence} confidence
                            </div>
                            <div style="font-style: italic; color: #666;">
                                Based on {model_type} model analysis of {chosen_coin}
                            </div>
                        </div>
                    </div>
                    """
                    
                    st.markdown(result_html, unsafe_allow_html=True)
                    st.caption("Note: This forecast is an estimate and market conditions can change unexpectedly.")
                else:
                    st.error("Unable to forecast price.")
    
# getting best coins   
def find_best_coins(model_type, desired_profit, num_days):
    if selected_data.empty:
        st.error("No selected coins data available.")
        return
    
    coins = selected_data.columns[:4]
    models = {}
    predictions = {}
    
    # Load models with better error handling
    for coin_index, coin in enumerate(coins, start=1):
        model_folder = f"trained_models/Model_SELECTED_COIN_{coin_index}"
        model_file = f"{model_folder}/{model_type.lower()}_model.pkl"
        
        try:
            if model_type == 'LSTM':
                model_path = model_file.replace('.pkl', '.keras')
                if os.path.exists(model_path):
                    models[coin] = tf.keras.models.load_model(model_path)
            else:
                if os.path.exists(model_file):
                    models[coin] = joblib.load(model_file)
        except Exception as e:
            st.warning(f"Failed to load model for {coin}: {str(e)}")
    
    if not models:
        st.error("No models were loaded successfully.")
        return
    
    # Make predictions for all coins
    for coin, model in models.items():
        try:
            input_data = np.array([[num_days, 0, 0]])
            
            if model_type == 'LSTM':
                input_data = input_data.reshape(1, input_data.shape[1], 1)
                price_change = model.predict(input_data)[0][0]
            else:
                price_change = model.predict(input_data)[0]
            
            potential_profit = price_change * desired_profit
            predictions[coin] = potential_profit
        except Exception as e:
            st.warning(f"Error predicting for {coin}: {str(e)}")
    
    if not predictions:
        st.error("No valid predictions could be generated.")
        return
    
    # Separate coins that exceed target from those that don't
    exceed_target = {coin: profit for coin, profit in predictions.items() if profit >= desired_profit}
    below_target = {coin: profit for coin, profit in predictions.items() if profit < desired_profit}
    
    # Sort coins that exceed target by profit (descending)
    # Sort coins below target by how close they are to target (ascending)
    
    recommended_coins = []
    
    # If we have coins that exceed target, recommend the best ones
    if exceed_target:
        sorted_exceed = sorted(exceed_target.items(), key=lambda x: x[1], reverse=True)
        # Take best performing and closest to target from those exceeding
        if len(sorted_exceed) >= 2:
            # Best performer
            recommended_coins.append(sorted_exceed[0])
            # Find closest to target but still exceeding
            sorted_by_closeness = sorted(exceed_target.items(), key=lambda x: abs(x[1] - desired_profit))
            recommended_coins.append(sorted_by_closeness[0])
        else:
            recommended_coins.append(sorted_exceed[0])
            # If we only have one coin exceeding, get the best from below target
            if below_target:
                sorted_below = sorted(below_target.items(), key=lambda x: abs(x[1] - desired_profit))
                recommended_coins.append(sorted_below[0])
    else:
        # If no coins exceed target, get the two closest
        sorted_below = sorted(below_target.items(), key=lambda x: abs(x[1] - desired_profit))
        if len(sorted_below) >= 2:
            recommended_coins.append(sorted_below[0])
            recommended_coins.append(sorted_below[1])
        elif len(sorted_below) == 1:
            recommended_coins.append(sorted_below[0])
    
    # Display results with improved formatting
    st.subheader("Results:")
    
    # First recommendation
    if len(recommended_coins) >= 1:
        coin, profit = recommended_coins[0]
        if profit >= desired_profit:
            st.success(f"Best performing coin: {coin}")
            exceeds_by = profit - desired_profit
            st.write(f"Predicted profit: ${profit:.2f} (exceeds  target  by ${exceeds_by:.2f})")
        else:
            st.warning(f"Closest coin: {coin}")
            shortfall = desired_profit - profit
            st.write(f"Predicted profit: ${profit:.2f} (below  target  by ${shortfall:.2f})")
        
        st.write(f"Time period: {num_days} days")
        profit_percentage = (profit / desired_profit) * 100 if desired_profit != 0 else 0
        st.write(f"Achieves {profit_percentage:.1f}% of desired profit (${desired_profit:.2f})")
    
    # Second recommendation
    if len(recommended_coins) >= 2:
        st.write("---")
        coin, profit = recommended_coins[1]
        if profit >= desired_profit:
            label = "Alternative option (exceeds target):"
            if profit > recommended_coins[0][1]:
                label = "Alternative option (higher profit):"
            elif abs(profit - desired_profit) < abs(recommended_coins[0][1] - desired_profit):
                label = "Alternative option (closer to target):"
            st.success(f"{label} {coin}")
        else:
            st.warning(f"Alternative option: {coin}")
        
        st.write(f"Predicted profit: ${profit:.2f}")
        st.write(f"Time period: {num_days} days")
        profit_percentage = (profit / desired_profit) * 100 if desired_profit != 0 else 0
        st.write(f"Achieves {profit_percentage:.1f}% of desired profit (${desired_profit:.2f})")

def get_top_crypto_news(crypto, num_stories=5, news_source='all'):
    if news_source == 'all' or news_source == 'Cryptoslate':
        cryptoslate_feed = feedparser.parse('https://cryptoslate.com/feed/')
        crypto_news = [entry for entry in cryptoslate_feed.entries if crypto.lower() in entry.title.lower()]
        
        if crypto_news:
            st.write(f"Top Cryptoslate news about {crypto}:")
            for i, entry in enumerate(crypto_news[:num_stories], start=1):
                st.write(f"{i}. [{entry.title}]({entry.link}) - {entry.published}")
    
    if news_source == 'all' or news_source == 'CoinDesk':
        coindesk_feed = feedparser.parse('https://feeds.feedburner.com/CoinDesk')
        crypto_news = [entry for entry in coindesk_feed.entries if crypto.lower() in entry.title.lower()]
        
        if crypto_news:
            st.write(f"Top CoinDesk news about {crypto}:")
            for i, entry in enumerate(crypto_news[:num_stories], start=1):
                st.write(f"{i}. [{entry.title}]({entry.link}) - {entry.published}")

# App styling
st.markdown("""
<style>
body {
    font-family: 'Arial', sans-serif;
    color: #ffffff;
    background-color: #4B0082;
    font-size: 18px;
}

.sidebar .sidebar-content {
    background-image: linear-gradient(#6a0dad, #7b68ee);
    color: #ffffff;
}

.sidebar .sidebar-content .Widget>label {
    color: #ffffff;
}

div.stButton > button:first-child {
    background-color: #6a0dad;
    color: #ffffff;
}

.reportview-container .main {
    background-color: #4B0082;
    color: #ffffff;
}

footer {
    background-color: #6a0dad;
    color: #ffffff;
}

header {
    background-color: #4B0082;
}

[data-testid="stAppViewContainer"] {
    background-image: url("https://img.freepik.com/premium-photo/white-bitcoin-cryptocurrency-coin-against-grey-background-d-rendering_601748-3477.jpg");
    background-size: cover;
}
</style>
""", unsafe_allow_html=True)

# Main app navigation
def main():
    initialize_session_state()
    check_versions()
    
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "About Us", "Dataset", "Coin Correlation", 
                                     "Moving Average", "Visualizations", "Predictions", "NEWS"])

    if page == "Home":
        home_section()
    elif page == "About Us":
        about_us()
    elif page == "Dataset":
        dataset_section()
    elif page == "Coin Correlation":
        analyze_coin_correlation()
    elif page == "Moving Average":
        plot_moving_average()
    elif page == "Visualizations":
        visualization_option = st.sidebar.radio("Select Visualization:", 
                                              ['Home', 'Price Comparison', 'Candlestick Chart',
                                               'Market State Visualization', "Predicted Highs and Lows"])
        
        if visualization_option == 'Home':
            st.title("Data Visualization")
            st.image('https://img.freepik.com/free-vector/gradient-stock-market-concept_23-2149166910.jpg', 
                    use_container_width=True)
            st.write("Explore different visualizations to gain insights into cryptocurrency markets.")
        elif visualization_option == 'Price Comparison':
            vis_option = st.sidebar.radio("Compare:", ['Metrics', 'Coins'])
            
            if vis_option == 'Metrics':
                coin = st.selectbox("Select cryptocurrency:", combined_data['Crypto'].unique())
                metrics = st.multiselect("Select metrics:", ['Close', 'Open', 'High', 'Low'])
                start_date = st.date_input("Start date:", datetime.now() - timedelta(days=365))
                end_date = st.date_input("End date:", datetime.now())
                
                if st.button("Plot"):
                    plot_crypto_metrics(coin, metrics, str(start_date), str(end_date))
            else:
                coins = st.multiselect("Select cryptocurrencies:", combined_data['Crypto'].unique())
                metric = st.selectbox("Select metric:", ['Close', 'Open', 'High', 'Low'])
                start_date = st.date_input("Start date:", datetime.now() - timedelta(days=365))
                end_date = st.date_input("End date:", datetime.now())
                
                if st.button("Plot"):
                    plot_crypto_coins(coins, metric, str(start_date), str(end_date))
        elif visualization_option == 'Candlestick Chart':
            coin = st.selectbox("Select cryptocurrency:", combined_data['Crypto'].unique())
            period = st.radio("Select period:", ['Daily', 'Weekly', 'Monthly'])
            plot_candlestick_chart(coin, period[0])
        elif visualization_option == 'Market State Visualization':
            visualize_market_state()
        elif visualization_option == "Predicted Highs and Lows":
            predict_highs_lows()
    elif page == "Predictions":
        prediction_option = st.sidebar.radio("Select:", 
                                           ["Dataset", "Training", "Training Model Metrics", "Prediction Graphs",
                                            "Buy and Sell Prediction", "Predict coin by Profit"])
        
        if prediction_option == "Dataset":
            display_selected_coins()
            plot_coin_scatter()
    
        elif prediction_option == "Training":
            st.header("Model Training")
            
            # Initialize session state variables if they don't exist
            if 'training_started' not in st.session_state:
                st.session_state.training_started = False
            if 'models_trained' not in st.session_state:
                st.session_state.models_trained = False
            if 'training_progress' not in st.session_state:
                st.session_state.training_progress = 0
            if 'total_models' not in st.session_state:
                st.session_state.total_models = 4  # Update this with your actual number of models

            # Check if training is complete
            if st.session_state.models_trained:
                st.success("All models trained successfully!")
                if st.button("Reset Training Status"):
                    st.session_state.training_started = False
                    st.session_state.models_trained = False
                    st.session_state.training_progress = 0
                    st.rerun()
            
            # Check if training is in progress
            elif st.session_state.training_started:
                # Show progress bar
                progress = st.session_state.training_progress / st.session_state.total_models
                st.progress(progress)
                
                if st.session_state.training_progress >= st.session_state.total_models:
                    st.session_state.models_trained = True
                    st.session_state.training_started = False
                    st.rerun()
                else:
                    st.warning(f"Training in progress... ({st.session_state.training_progress}/{st.session_state.total_models} models completed)")
                    if st.button("Refresh Status"):
                        st.rerun()
            
            # Initial state - no training started yet
            else:
                if st.button("Train All Models"):
                    st.session_state.training_started = True
                    st.session_state.training_progress = 0
                    thread = threading.Thread(
                        target=train_all_models_background,
                        args=(selected_data,)
                    )
                    thread.start()
                    st.rerun()
        
        elif prediction_option == "Training Model Metrics":
            coins = st.multiselect("Select coins:", selected_data.columns)
            model = st.selectbox("Select model:", ['all', 'Gradient Boosting', 'SVR', 'XGBoost', 'LSTM'])
            
            for coin in coins:
                coin_index = selected_data.columns.get_loc(coin)
                evaluate_models_selected_coin(selected_data, coin_index, model)
        elif prediction_option == "Prediction Graphs":
            coin = st.selectbox("Select coin:", selected_data.columns)
            model = st.selectbox("Select model:", ['Gradient Boosting', 'SVR', 'XGBOOST', 'LSTM'])
            frequency = st.selectbox("Select frequency:", ['daily', 'weekly', 'monthly', 'quarterly'])
            periods = st.number_input("Number of periods:", min_value=1, value=20)
            
            if st.button("Predict"):
                coin_index = selected_data.columns.get_loc(coin)
                evaluate_and_plot_model(coin_index, model, frequency, periods)
        elif prediction_option == "Buy and Sell Prediction":
            strategy = st.sidebar.radio(
                "Select Prediction Strategy:",
                ["Moving Averages", "Machine Learning Models"],
                help="Choose between technical indicators or AI models for predictions"
            )
            
            st.markdown("## Buy/Sell Recommendation Prediction")
            
            if strategy == "Moving Averages":
                col1, col2 = st.columns([2, 1])
                with col1:
                    coin = st.selectbox(
                        "Select Cryptocurrency:", 
                        selected_data.columns,
                        key="ma_coin_select"
                    )
                with col2:
                    days = st.slider(
                        "Forecast Period (days):",
                        min_value=1, 
                        max_value=30,
                        value=10,
                        key="ma_days_slider"
                    )
                
                if st.button("Generate Prediction", key="ma_predict_btn", 
                            help="Generate buy/sell recommendation based on moving averages"):
                    with st.spinner("Analyzing market trends with Moving Averages..."):
                        determine_best_time_to_trade_future(coin, days)
            
            else:  # Machine Learning Models
                create_prediction_interface(selected_data)
        elif prediction_option == "Predict coin by Profit":
            model_type = st.selectbox("Select model:", ['Gradient_Boosting', 'SVR', 'Xgboost', 'LSTM'])
            profit = st.number_input("Desired profit:", value=100)
            days = st.number_input("Days:", value=30)
            
            if st.button("Find Best Coins"):
                find_best_coins(model_type, profit, days)
    elif page == "NEWS":
        crypto = st.text_input("Cryptocurrency:", "Bitcoin")
        source = st.selectbox("News source:", ['all', 'Cryptoslate', 'CoinDesk'])
        get_top_crypto_news(crypto, news_source=source)

if __name__ == "__main__":
    main()



    
   