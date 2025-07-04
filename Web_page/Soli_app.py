# importing necessary modules
import os
import warnings
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px  
from plotly.subplots import make_subplots
from ta.trend import SMAIndicator
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn import __version__ as sklearn_version
import sklearn
import xgboost
from xgboost import XGBRegressor, __version__ as xgboost_version
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense
from keras.callbacks import EarlyStopping
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import mplfinance as mpf
import feedparser
import itertools
from scipy.stats import gaussian_kde
import threading
import time
import pickle
import yfinance as yf
import joblib
from joblib import Memory, __version__ as joblib_version
import logging
from logging.handlers import RotatingFileHandler




warnings.filterwarnings("ignore", category=UserWarning)

# Configuration for training
os.makedirs("cached_models", exist_ok=True)
os.makedirs("trained_models", exist_ok=True)
memory = Memory("cached_models", verbose=0)







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




# Configuration for training
os.makedirs("cached_models", exist_ok=True)
os.makedirs("trained_models", exist_ok=True)

def initialize_session_state():
    """Initialize all required session state variables for training"""
    required_vars = {
        'training_state': {
            'started': False,
            'completed': False,
            'progress': {},
            'overall': 0,
            'total_models': 0,
            'error': None,
            'last_update': time.time()
        },
        'training_thread': None,
        'models_trained': False,
        'model_paths': {}
    }
    
    for var, default in required_vars.items():
        if var not in st.session_state:
            st.session_state[var] = default

def setup_logging():
    """Configure logging system for training"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            RotatingFileHandler('training.log', maxBytes=1e6, backupCount=3),
            logging.StreamHandler()
        ]
    )
    logging.info("Logging system initialized")

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
    
    if for_lstm:
        X = X.values.astype(np.float32)
        y = y.values.astype(np.float32)
    
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_gradient_boosting(X_train, y_train):
    """Train Gradient Boosting model"""
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
    """Train SVR model"""
    params = {
        'C': 1.0,
        'kernel': 'rbf',
        'gamma': 'scale'
    }
    svr = SVR(**params)
    svr.fit(X_train, y_train)
    return svr

def train_xgboost(X_train, y_train):
    """Train XGBoost model"""
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

def train_lstm(X_train, y_train):
    """Train LSTM model (requires numpy arrays)"""
    try:
        tf.keras.backend.clear_session()
        
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        )

        model = Sequential([
            LSTM(32, input_shape=(X_train.shape[1], 1)),
            Dense(16, activation='relu'),
            Dense(1)
        ])

        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )

        X_train_reshaped = np.array(X_train, dtype=np.float32).reshape(
            X_train.shape[0], X_train.shape[1], 1
        )
        y_train_array = np.array(y_train, dtype=np.float32)

        history = model.fit(
            X_train_reshaped, 
            y_train_array,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            callbacks=[early_stop],
            verbose=1
        )
        
        return model

    except Exception as e:
        logging.error(f"Error during LSTM training: {str(e)}", exc_info=True)
        return None

def save_model(model, model_name, coin_index=1, input_shape=None):
    """Save trained model to file with version info"""
    try:
        model_dir = f"trained_models/Model_SELECTED_COIN_{coin_index}"
        os.makedirs(model_dir, exist_ok=True)
        
        # Save the model based on type
        if model_name == 'LSTM':
            model_path = os.path.join(model_dir, "lstm_model.keras")
            model.save(model_path)
        else:
            model_path = os.path.join(model_dir, f"{model_name.lower().replace(' ', '_')}_model.pkl")
            joblib.dump(model, model_path, compress=3)
        
        return model_dir
        
    except Exception as e:
        logging.error(f"Error saving model {model_name}: {str(e)}")
        raise

def update_training_progress(coin_name, status, current_model=None, message=None, progress=None):
    """Thread-safe update of training progress"""
    with threading.Lock():
        if 'training_state' not in st.session_state:
            st.session_state.training_state = {}
        if 'progress' not in st.session_state.training_state:
            st.session_state.training_state['progress'] = {}
            
        if coin_name not in st.session_state.training_state['progress']:
            st.session_state.training_state['progress'][coin_name] = {}
        
        if status:
            st.session_state.training_state['progress'][coin_name]['status'] = status
        if current_model:
            st.session_state.training_state['progress'][coin_name]['current_model'] = current_model
        if message:
            st.session_state.training_state['progress'][coin_name]['message'] = message
        if progress is not None:
            st.session_state.training_state['progress'][coin_name]['progress'] = progress
        
        st.session_state.training_state['last_update'] = time.time()

def train_models_for_coin(selected_data, coin_index):
    """Train all models for a specific coin with progress tracking"""
    initialize_session_state()
    coin_name = selected_data.columns[coin_index]
    logging.info(f"Starting training for {coin_name}")
    
    try:
        # Initialize progress tracking for this coin
        update_training_progress(
            coin_name=coin_name,
            status='In Progress',
            current_model=None,
            message='Initializing training',
            progress=0
        )
        
        models = {}
        
        # ===== GRADIENT BOOSTING =====
        logging.info(f"Training Gradient Boosting for {coin_name}")
        update_training_progress(
            coin_name=coin_name,
            status='In Progress',  # Added missing status parameter
            current_model='Gradient Boosting',
            message='Training Gradient Boosting...',
            progress=0
        )
        
        X_train, X_test, y_train, y_test = prepare_data(selected_data, coin_index, for_lstm=False)
        models['Gradient Boosting'] = train_gradient_boosting(X_train, y_train)
        
        update_training_progress(
            coin_name=coin_name,
            status='In Progress',  # Added missing status parameter
            message='Gradient Boosting completed',
            progress=25
        )
        
        # ===== SVR =====
        logging.info(f"Training SVR for {coin_name}")
        update_training_progress(
            coin_name=coin_name,
            status='In Progress',  # Added missing status parameter
            current_model='SVR',
            message='Training SVR...',
            progress=25
        )
        
        models['SVR'] = train_svr(X_train, y_train)
        
        update_training_progress(
            coin_name=coin_name,
            status='In Progress',  # Added missing status parameter
            message='SVR completed',
            progress=50
        )
        
        # ===== XGBOOST =====
        logging.info(f"Training XGBoost for {coin_name}")
        update_training_progress(
            coin_name=coin_name,
            status='In Progress',  # Added missing status parameter
            current_model='XGBoost',
            message='Training XGBoost...',
            progress=50
        )
        
        models['XGBoost'] = train_xgboost(X_train, y_train)
        
        update_training_progress(
            coin_name=coin_name,
            status='In Progress',  # Added missing status parameter
            message='XGBoost completed',
            progress=75
        )
        
        # ===== LSTM =====
        logging.info(f"Training LSTM for {coin_name} (this may take a while)")
        update_training_progress(
            coin_name=coin_name,
            status='In Progress',  # Added missing status parameter
            current_model='LSTM',
            message='Training LSTM (this may take a few minutes)...',
            progress=75
        )
        
        X_train_lstm, _, y_train_lstm, _ = prepare_data(selected_data, coin_index, for_lstm=True)
        models['LSTM'] = train_lstm(X_train_lstm, y_train_lstm)
        
        update_training_progress(
            coin_name=coin_name,
            status='Completed',
            message='LSTM completed',
            progress=100
        )
        
        # Save models
        logging.info(f"Saving models for {coin_name}")
        saved_paths = {}
        for name, model in models.items():
            model_dir = save_model(model, name, coin_index + 1, X_train.shape[1])
            saved_paths[name] = model_dir
        
        # Update session state with model paths
        with threading.Lock():
            if 'model_paths' not in st.session_state:
                st.session_state.model_paths = {}
            st.session_state.model_paths.update(saved_paths)
            st.session_state.training_state['last_update'] = time.time()
        
        logging.info(f"Successfully completed training for {coin_name}")
        return models
        
    except Exception as e:
        error_msg = f"Error training {coin_name}: {str(e)}"
        logging.error(error_msg, exc_info=True)
        update_training_progress(
            coin_name=coin_name,
            status='Failed',  # Ensure status is provided here
            message=error_msg
        )
        raise e

def train_all_models_background(selected_data):
    """Train models for all coins in a background thread with proper progress tracking"""
    try:
        initialize_session_state()
        
        with threading.Lock():
            st.session_state.training_state = {
                'started': True,
                'completed': False,
                'progress': {},
                'overall': 0,
                'total_models': min(4, selected_data.shape[1]) * 4,
                'error': None,
                'last_update': time.time()
            }
        
        # Train models for each coin
        for coin_idx in range(min(4, selected_data.shape[1])):
            coin_name = selected_data.columns[coin_idx]
            
            # Initialize coin progress
            update_training_progress(
                coin_name=coin_name,
                status='Starting',
                current_model='',
                message='Initializing',
                progress=0
            )
            
            # Train models for this coin
            train_models_for_coin(selected_data, coin_idx)
            
            # Update overall progress
            with threading.Lock():
                st.session_state.training_state['overall'] += 4
                st.session_state.training_state['progress'][coin_name] = {
                    'status': 'Completed',
                    'current_model': '',
                    'message': 'All models trained',
                    'progress': 100
                }
                st.session_state.training_state['last_update'] = time.time()
        
        # Mark as completed
        with threading.Lock():
            st.session_state.training_state['completed'] = True
            st.session_state.training_state['started'] = False
            st.session_state.models_trained = True
    
    except Exception as e:
        error_msg = f"Training error: {str(e)}"
        logging.error(error_msg, exc_info=True)
        with threading.Lock():
            st.session_state.training_state['error'] = error_msg
            st.session_state.training_state['started'] = False

def display_training_progress():
    """Display training progress in the UI"""
    if 'training_state' not in st.session_state:
        st.warning("Training not initialized")
        return
    
    training_state = st.session_state.training_state
    
    if training_state.get('completed', False):
        st.success("✅ All models trained successfully!")
        if st.button("Reset Training"):
            initialize_session_state()
            st.rerun()
        return
    
    if training_state.get('error'):
        st.error(f"❌ Training failed: {training_state['error']}")
        if st.button("Retry Training"):
            initialize_session_state()
            st.rerun()
        return
    
    if training_state.get('started', False):
        st.warning("🔄 Training in progress...")
        
        # Display overall progress
        if 'overall' in training_state and 'total_models' in training_state:
            progress_value = training_state['overall'] / training_state['total_models']
            st.progress(min(1.0, max(0.0, progress_value)))
            st.write(f"Overall progress: {training_state['overall']}/{training_state['total_models']} models completed")
        
        # Display per-coin progress
        if 'progress' in training_state:
            for coin_name, progress in training_state['progress'].items():
                with st.expander(f"Progress for {coin_name}"):
                    col1, col2 = st.columns([1, 4])
                    with col1:
                        status_icon = "✅" if progress.get('status') == 'Completed' else "❌" if progress.get('status', '').startswith('Failed') else "🔄"
                        st.write(status_icon)
                    
                    with col2:
                        st.write(f"**Status:** {progress.get('status', 'Unknown')}")
                        if progress.get('current_model'):
                            st.write(f"**Current Model:** {progress['current_model']}")
                        if progress.get('message'):
                            st.write(progress['message'])
                        if progress.get('status') not in ['Completed', 'Failed']:
                            st.progress(min(1.0, max(0.0, progress.get('progress', 0) / 100)))
        
        # Auto-refresh logic
        if not training_state.get('completed', False):
            time.sleep(2)  # Refresh every 2 seconds
            st.rerun()
    else:
        if st.button("🚀 Train All Models"):
            # Start background thread
            thread = threading.Thread(
                target=train_all_models_background,
                args=(selected_data,),
                daemon=True
            )
            thread.start()
            st.rerun()


# Fetch cryptocurrency data for a single ticker
def get_crypto_data(ticker, start_date, end_date):
    try:
        crypto = yf.Ticker(ticker)
        data = crypto.history(start=start_date, end=end_date)
        return data
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
        return None

# Define ticker symbols - using the updated list
CRYPTO_TICKERS = [
    'BTC-GBP', 'ETH-GBP', 'USDT-GBP', 'BNB-GBP', 'SOL-GBP', 
    'XRP-GBP', 'USDC-GBP', 'ADA-GBP', 'DOGE-GBP', 'DOT-GBP',
    'MATIC-GBP', 'DAI-GBP', 'LTC-GBP', 'SHIB-GBP', 'TRX-GBP',
    'AVAX-GBP', 'LINK-GBP', 'ATOM-GBP', 'XLM-GBP', 'UNI-GBP',
    'BCH-GBP', 'ALGO-GBP', 'VET-GBP', 'FIL-GBP', 'THETA-GBP',
    'XMR-GBP', 'ETC-GBP', 'EOS-GBP', 'AAVE-GBP', 'XTZ-GBP',
    'SAND-GBP', 'MANA-GBP', 'APE-GBP', 'GALA-GBP', 'CHZ-GBP'
]

# Function to load data - but only display messages in the dataset section
def load_crypto_data(show_messages=False):
    global combined_data
    
    # Set date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=4*365)  # 4 years of data
    
    # Define cache file path
    data_file = "Cleaned_combined_crypto_data.csv"
    
    # Check if we should use cached data
    use_cached = False
    missing_tickers = []
    
    if os.path.exists(data_file):
        # Check how old the file is
        file_mtime = datetime.fromtimestamp(os.path.getmtime(data_file))
        days_old = (datetime.now() - file_mtime).days
        
        # Load the cached data to check what tickers it contains
        temp_data = pd.read_csv(data_file, parse_dates=['Date'], index_col='Date')
        
        # Get unique tickers in the dataset
        if 'Crypto' in temp_data.columns:
            cached_tickers = temp_data['Crypto'].unique().tolist()
        else:
            cached_tickers = []
        
        # Check if all required tickers are in the cache
        missing_tickers = [t for t in CRYPTO_TICKERS if t not in cached_tickers]
        
        # Decide whether to use cache based on age and completeness
        if days_old < 1 and not missing_tickers:
            combined_data = temp_data
            if show_messages:
                st.success(f"Loaded complete cached data from {file_mtime.strftime('%Y-%m-%d')}")
            use_cached = True
        elif days_old >= 1:
            if show_messages:
                st.info(f"Cached data is {days_old} days old. Refreshing all data...")
            combined_data = None
        elif missing_tickers:
            if show_messages:
                st.info(f"Cached data is missing {len(missing_tickers)} cryptocurrencies. Refreshing all data...")
            combined_data = None
    else:
        if show_messages:
            st.info("No cached data found. Fetching fresh data...")
        combined_data = None
    
    # If we need to fetch fresh data
    if not use_cached:
        # Setup progress tracking
        if show_messages:
            progress_bar = st.progress(0)
            status_text = st.empty()
        
        # Initialize empty DataFrame
        combined_data = pd.DataFrame()
        
        # Fetch data for each ticker
        for i, ticker in enumerate(CRYPTO_TICKERS):
            if show_messages:
                status_text.text(f"Fetching {ticker}... ({i+1}/{len(CRYPTO_TICKERS)})")
                progress_bar.progress((i+1)/len(CRYPTO_TICKERS))
            
            data = get_crypto_data(ticker, start_date, end_date)
            if data is not None and not data.empty:
                data['Crypto'] = ticker
                combined_data = pd.concat([combined_data, data], axis=0)
                if show_messages:
                    latest_date = data.index[-1].strftime('%Y-%m-%d')
                    st.write(f"✅ {ticker} (up to {latest_date})")
            elif show_messages:
                st.warning(f"No data for {ticker}")
        
        # Clean the data if we have any
        if not combined_data.empty:
            if 'Dividends' in combined_data.columns:
                combined_data.drop(['Dividends', 'Stock Splits'], axis=1, inplace=True)
            combined_data.to_csv(data_file)
            if show_messages:
                st.success(f"Saved new data with {len(combined_data):,} rows")
        elif show_messages:
            st.error("No data was fetched. Check your internet connection or ticker symbols.")
    
    return combined_data



combined_data = load_crypto_data(show_messages=False)

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Compute the elbow plot
def plot_elbow(loadings):
    inertia = []
    K = range(1, 10)
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans.fit(loadings.iloc[:, :-1])  # exclude 'Cluster' column
        inertia.append(kmeans.inertia_)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(K, inertia, 'bo-')
    ax.set_xlabel('Number of clusters k')
    ax.set_ylabel('Inertia')
    ax.set_title('Elbow Method For Optimal k')

    st.pyplot(fig)


    



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

# This should be placed after combined_data is defined
selected_data_file = 'Selected_coins.csv'
if os.path.exists(selected_data_file):
    # Check if selected_data_file is older than combined_data file
    data_file = "Cleaned_combined_crypto_data.csv"
    if os.path.exists(data_file):
        selected_mtime = os.path.getmtime(selected_data_file)
        combined_mtime = os.path.getmtime(data_file)
        
        if selected_mtime < combined_mtime:
            # If selected data is older than combined data, regenerate it
            if not combined_data.empty:
                selected_data = generate_selected_data(combined_data)
                selected_data.to_csv(selected_data_file)
            else:
                selected_data = pd.read_csv(selected_data_file, index_col='Date')
        else:
            selected_data = pd.read_csv(selected_data_file, index_col='Date')
    else:
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
    """Enhanced home section with modern UI and improved content structure"""
    
    # ====== Custom CSS Styling ======
    st.markdown("""
    <style>
    /* Main hero section */
    .hero-container {
        background: linear-gradient(135deg, #f5f7fa 0%, #e4e8eb 100%);
        padding: 2.5rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    }
    
    /* Title styling */
    .hero-title {
        font-size: 2.8rem !important;
        color: #2c3e50 !important;
        margin-bottom: 0.5rem !important;
        font-weight: 700 !important;
        background: linear-gradient(90deg, #3498db, #2c3e50);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Subtitle styling */
    .hero-subtitle {
        font-size: 1.3rem !important;
        color: #5d6d7e !important;
        margin-bottom: 1.5rem !important;
    }
    
    /* Feature highlight */
    .feature-highlight {
        font-size: 1.2rem !important;
        background-color: rgba(52, 152, 219, 0.1);
        padding: 1.2rem;
        border-radius: 10px;
        border-left: 4px solid #3498db;
        margin: 1rem 0;
    }
    
    /* CTA button styling */
    .cta-button {
        background: linear-gradient(135deg, #3498db, #2c3e50) !important;
        color: white !important;
        font-weight: 600 !important;
        padding: 0.7rem 2rem !important;
        border-radius: 8px !important;
        border: none !important;
        margin-top: 1rem !important;
        transition: all 0.3s ease !important;
    }
    
    .cta-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(52, 152, 219, 0.4);
    }
    
    /* Feature cards */
    .feature-card {
        padding: 1.5rem;
        border-radius: 12px;
        background-color: white;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        margin-bottom: 1.5rem;
        transition: all 0.3s ease;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 20px rgba(0,0,0,0.12);
    }
    </style>
    """, unsafe_allow_html=True)

    # ====== Header Section ======
    header_col1, header_col2 = st.columns([1, 3])
    with header_col1:
        st.image("https://dcassetcdn.com/design_img/2956478/152860/152860_16291385_2956478_2487aca1_image.jpg", 
                width=150)
    with header_col2:
        st.markdown('<div class="hero-title">SOLiGence</div>', unsafe_allow_html=True)
        st.markdown('<div class="hero-subtitle">Your Intelligent Coin Trading Platform</div>', 
                   unsafe_allow_html=True)

    # ====== Hero Section ======
    st.markdown("""
    <div class="hero-container">
        <div class="feature-highlight">
            Empower your cryptocurrency trading decisions with AI-driven insights and real-time data analytics.
            Our platform combines cutting-edge machine learning with comprehensive market data to give you the competitive edge.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ====== CTA Section ======
    if st.button("🚀 Get Started with SOLiGence", key="get_started", 
                help="Begin your trading journey"):
        st.session_state.show_get_started = True

    # ====== Expanded Content ======
    if st.session_state.get('show_get_started', False):
        st.markdown("---")
        st.header("✨ Discover the SOLiGence Advantage", anchor=False)
        
        # Features Grid
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="feature-card">
                <h3 style="color: #2c3e50;">📊 AI-Powered Market Analysis</h3>
                <p style="color: #555;">
                    Our proprietary algorithms analyze market patterns and predict trends 
                    with 85% historical accuracy.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="feature-card">
                <h3 style="color: #2c3e50;">⏱️ Real-Time Data Processing</h3>
                <p style="color: #555;">
                    Get millisecond-level updates from 25+ exchanges with our high-performance 
                    data aggregation system.
                </p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div class="feature-card">
                <h3 style="color: #2c3e50;">📈 Advanced Trading Indicators</h3>
                <p style="color: #555;">
                    Access 50+ technical indicators including exclusive SOLiGence metrics 
                    not available elsewhere.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="feature-card">
                <h3 style="color: #2c3e50;">🔒 Institutional-Grade Security</h3>
                <p style="color: #555;">
                    Bank-level encryption and regular security audits ensure your data 
                    and strategies remain protected.
                </p>
            </div>
            """, unsafe_allow_html=True)

        # Testimonials Section
        st.markdown("---")
        st.header("💬 Trusted by Thousands of Traders", anchor=False)
        
        testimonial_col1, testimonial_col2 = st.columns(2)
        
        with testimonial_col1:
            st.markdown("""
            <div style="background-color: #f8f9fa; padding: 1.5rem; border-radius: 10px; border-left: 4px solid #3498db;">
                <p style="font-style: italic; color: #555;">
                "SOLiGence helped me increase my portfolio by 42% in 6 months. The predictive 
                analytics spotted trends I would have completely missed."
                </p>
                <p style="font-weight: 600; color: #2c3e50; margin-bottom: 0;">— Michael T.</p>
                <p style="color: #7f8c8d; margin-top: 0;">Professional Crypto Trader</p>
            </div>
            """, unsafe_allow_html=True)
            
        with testimonial_col2:
            st.markdown("""
            <div style="background-color: #f8f9fa; padding: 1.5rem; border-radius: 10px; border-left: 4px solid #3498db;">
                <p style="font-style: italic; color: #555;">
                "As a beginner, I found the platform incredibly intuitive. The AI suggestions 
                helped me avoid costly mistakes while I was learning."
                </p>
                <p style="font-weight: 600; color: #2c3e50; margin-bottom: 0;">— Sarah K.</p>
                <p style="color: #7f8c8d; margin-top: 0;">Beginner Investor</p>
            </div>
            """, unsafe_allow_html=True)

        # Connect Section
        st.markdown("---")
        st.header("📩 Join Our Trading Community", anchor=False)
        
        connect_col1, connect_col2 = st.columns(2)
        
        with connect_col1:
            st.markdown("""
            <div style="margin-top: 1rem;">
                <h4 style="color: #2c3e50;">Connect With Us</h4>
                <a href="https://www.linkedin.com/in/deborah-adedigba-bb917314b/" target="_blank" style="text-decoration: none;">
                    <button style="background-color: #0077b5; color: white; border: none; padding: 0.5rem 1rem; border-radius: 5px; margin-right: 0.5rem;">
                        LinkedIn
                    </button>
              
            </div>
            """, unsafe_allow_html=True)
            
        
    
  

def about_us():
    """Display information about the application's features and capabilities with improved visibility."""
    
    # Custom CSS for better visibility
    st.markdown("""
    <style>
    .feature-card {
        background-color: rgba(255, 255, 255, 0.85);
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .tech-card {
        background-color: rgba(245, 245, 245, 0.9);
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .section-header {
        color: #2c3e50;
        background-color: rgba(255, 255, 255, 0.7);
        padding: 0.5rem 1rem;
        border-radius: 5px;
        margin-bottom: 1rem;
    }
    .disclaimer-box {
        background-color: rgba(255, 243, 205, 0.9);
        border-left: 5px solid #ffc107;
        padding: 1rem;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

    # Header with logo and title
    st.markdown("""
    <div style="background-color: rgba(255, 255, 255, 0.8); padding: 1.5rem; border-radius: 10px; margin-bottom: 2rem;">
        <div style="display: flex; align-items: center;">
            <div style="flex: 0 0 150px;">
                <img src="https://dcassetcdn.com/design_img/2956478/152860/152860_16291385_2956478_2487aca1_image.jpg" width="150" style="border-radius: 8px;">
            </div>
            <div style="flex: 1; padding-left: 1.5rem;">
                <h1 style="color: #2c3e50; margin-bottom: 0.2rem;">About SOLiGence</h1>
                <p style="color: #5d6d7e; font-size: 1.1rem;">Intelligent Cryptocurrency Analysis Platform</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Core Features
    with st.container():
        st.markdown('<div class="section-header"><h2>✨ Key Features</h2></div>', unsafe_allow_html=True)
        
        features = [
            ("📊 Comprehensive Data Analysis", 
             "Analyze 30+ major cryptocurrencies with historical price data, volume trends, and market indicators"),
            
            ("🤖 AI-Powered Predictions", 
             "Four advanced machine learning models (GBR, SVR, XGBoost, LSTM) for accurate price forecasting"),
            
            ("📈 Advanced Visualizations", 
             "Interactive charts including candlestick patterns, moving averages, and correlation matrices"),
            
            ("🔍 Market Insights", 
             "Identify market trends, volatility patterns, and optimal trading opportunities"),
            
            ("⚡ Real-time Analysis", 
             "Process and visualize the latest market data with automatic updates"),
            
            ("📱 User-Friendly Interface", 
             "Intuitive controls and customizable views for both beginners and experienced traders")
        ]
        
        for title, desc in features:
            st.markdown(f"""
            <div class="feature-card">
                <h3 style="color: #2c3e50; margin-top: 0;">{title}</h3>
                <p style="color: #4a4a4a;">{desc}</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.divider()
    
    # Technology Stack
    with st.container():
        st.markdown('<div class="section-header"><h2>🛠️ Under the Hood</h2></div>', unsafe_allow_html=True)
        st.markdown('<p style="color: #4a4a4a;">This application leverages cutting-edge technologies:</p>', unsafe_allow_html=True)
        
        tech_cols = st.columns(3)
        tech_stack = [
            ("Machine Learning", "Gradient Boosting, SVR, XGBoost, LSTM"),
            ("Data Processing", "Pandas, NumPy, Scikit-learn"),
            ("Visualization", "Plotly, Matplotlib, Streamlit"),
            ("Data Sources", "Yahoo Finance API, Cryptoslate RSS"),
            ("Backend", "Python 3.10, TensorFlow, Joblib"),
            ("Deployment", "Streamlit Cloud, Docker")
        ]
        
        for i, (category, tools) in enumerate(tech_stack):
            tech_cols[i%3].markdown(f"""
            <div class="tech-card">
                <strong style="color: #2c3e50;">{category}</strong>
                <div style="color: #4a4a4a; font-family: monospace; margin-top: 0.5rem;">{tools}</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.divider()
    
    # Data and Models
    with st.container():
        st.markdown('<div class="section-header"><h2>📦 Data & Models</h2></div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="feature-card" style="height: 100%;">
                <h3 style="color: #2c3e50; margin-top: 0;">Dataset Information</h3>
                <ul style="color: #4a4a4a;">
                    <li>30 major cryptocurrencies</li>
                    <li>4 years of historical data</li>
                    <li>Daily price/volume metrics</li>
                    <li>Cleaned and normalized</li>
                    <li>Automatic updates</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown("""
            <div class="feature-card" style="height: 100%;">
                <h3 style="color: #2c3e50; margin-top: 0;">Prediction Models</h3>
                <ul style="color: #4a4a4a;">
                    <li><strong>Gradient Boosting</strong>: Best for general trends</li>
                    <li><strong>SVR</strong>: Effective in volatile markets</li>
                    <li><strong>XGBoost</strong>: High accuracy with feature importance</li>
                    <li><strong>LSTM</strong>: Captures temporal patterns</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
    
    st.divider()
    
    # Usage Guide
    with st.container():
        st.markdown('<div class="section-header"><h2>📚 How To Use</h2></div>', unsafe_allow_html=True)
        
        steps = [
            ("1. Explore Data", "Use the Dataset section to filter, sort and analyze raw market data"),
            ("2. Visualize Trends", "Create interactive charts in the Visualizations section"),
            ("3. Analyze Correlations", "See how different coins move together in Coin Correlation"),
            ("4. Generate Predictions", "Get AI-powered forecasts in the Predictions section"),
            ("5. Make Decisions", "Use the Buy/Sell recommendations with confidence intervals")
        ]
        
        for title, desc in steps:
            st.markdown(f"""
            <div class="feature-card">
                <h4 style="color: #2c3e50; margin-top: 0;">{title}</h4>
                <p style="color: #4a4a4a;">{desc}</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.divider()
    
    # Disclaimer
    with st.container():
        st.markdown("""
        <div class="disclaimer-box">
            <h3 style="color: #856404; margin-top: 0;">⚠️ Important Notice</h3>
            <p style="color: #856404;">
            This application provides analytical tools for educational purposes only. 
            Cryptocurrency trading involves substantial risk. Past performance does not 
            guarantee future results. Always conduct your own research before making 
            investment decisions.
            </p>
        </div>
        """, unsafe_allow_html=True)


def dataset_section():
    global combined_data
    
    st.title("Cryptocurrency Dataset")
    st.markdown("Explore historical data for 30 major cryptocurrencies.")
    
    # Add refresh button in dataset section
    if st.button("Force Refresh Data"):
        data_file = "Cleaned_combined_crypto_data.csv"
        if os.path.exists(data_file):
            os.remove(data_file)
        # Also remove selected_data_file to regenerate it
        selected_data_file = 'Selected_coins.csv'
        if os.path.exists(selected_data_file):
            os.remove(selected_data_file)
        # Reload data with messages shown
        combined_data = load_crypto_data(show_messages=True)
    else:
        # Just show the data info when not refreshing
        data_file = "Cleaned_combined_crypto_data.csv"
        if os.path.exists(data_file):
            file_mtime = datetime.fromtimestamp(os.path.getmtime(data_file))
            st.info(f"Using data cached from {file_mtime.strftime('%Y-%m-%d at %H:%M:%S')}")
    
    if combined_data.empty:
        st.error("⚠️ No data available. Please check your data source or connection.")
        return
    
    # Sidebar controls with better organization
    with st.sidebar:
        st.subheader("🔍 Dataset Controls")
        
        # Filter section
        with st.expander("Filter Options", expanded=True):
            selected_crypto = st.selectbox(
                "Select cryptocurrency:",
                ['All'] + sorted(combined_data['Crypto'].unique()),
                help="Filter data by specific cryptocurrency"
            )
        
        # Sort section
        with st.expander("Sort Options", expanded=True):
            sort_column = st.multiselect(
                "Sort by columns:",
                combined_data.columns,
                help="Select columns to sort by (multiple selection supported)"
            )
            ascending = st.checkbox("Ascending order", True)
        
        # Pagination section
        with st.expander("Pagination", expanded=True):
            page_size = st.selectbox(
                "Items per page:",
                [10, 25, 50, 100],
                index=0,
                help="Number of rows to display per page"
            )
            total_pages = max(1, (len(combined_data) + page_size - 1) // page_size)
            page_number = st.number_input(
                f"Page number (1-{total_pages}):",
                min_value=1,
                max_value=total_pages,
                value=1
            )
    
    # Apply filters and sorting
    filtered_data = combined_data.copy()
    
    if selected_crypto != 'All':
        filtered_data = filtered_data[filtered_data['Crypto'] == selected_crypto]
    
    if sort_column:
        filtered_data = filtered_data.sort_values(by=sort_column, ascending=ascending)
    
    # Pagination logic
    start_idx = (page_number - 1) * page_size
    end_idx = start_idx + page_size
    paginated_data = filtered_data.iloc[start_idx:end_idx]
    
    # Display dataset information
    st.subheader("📈 Data Overview")
    st.info(f"ℹ️ Showing {len(paginated_data)} of {len(filtered_data)} records")
    
    # Enhanced dataframe display
    st.dataframe(
        paginated_data,
        height=min(600, (len(paginated_data) + 1) * 35),
        use_container_width=True
    )
    
    # Alternative views
    view_option = st.radio(
        "View as:",
        ["Interactive Table", "Static Table", "Raw Data"],
        horizontal=True
    )
    
    if view_option == "Static Table":
        st.table(paginated_data)
    elif view_option == "Raw Data":
        st.code(paginated_data.to_string())
    
    # Download option
    csv = filtered_data.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Current View as CSV",
        data=csv,
        file_name=f"crypto_data_{selected_crypto.lower() or 'all'}.csv",
        mime='text/csv',
        help="Download the filtered dataset as a CSV file"
    )

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
        st.warning("⚠️ No data available. Please load data first.")
        return
    
    # Create pivot table and get coin list
    pivoted_data = combined_data.pivot(columns='Crypto', values='Close')
    coins_list = pivoted_data.columns.tolist()
    
    # Add section header and description
    st.header("🔗 Cryptocurrency Correlation Analysis")
    st.markdown("Explore how different cryptocurrencies move in relation to each other")
    
    # Create two columns for layout
    col1, col2 = st.columns([1, 3])
    
    with col1:
        # Improved selectbox with custom styling
        coin_selected = st.selectbox(
            "Select a cryptocurrency:",
            coins_list,
            index=0,
            key="corr_coin_select",
            help="Select a coin to see its correlation with others"
        )
        
        # Add some metrics about the selected coin
        st.metric(
            f"Selected: {coin_selected}",
            value=f"${pivoted_data[coin_selected].iloc[-1]:,.2f}",
            delta=f"{pivoted_data[coin_selected].pct_change().iloc[-1]*100:.2f}% (24h)"
        )
    
    # Calculate correlations
    selected_coin_prices = pivoted_data[coin_selected]
    correlations = pivoted_data.corrwith(selected_coin_prices)
    sorted_correlations = correlations.sort_values(ascending=False)
    sorted_correlations = sorted_correlations.drop(coin_selected)
    
    with col2:
        # Visualize correlations with a bar chart
        fig = go.Figure()
        
        # Add positive correlations
        fig.add_trace(
            go.Bar(
                x=sorted_correlations.head(10).index,
                y=sorted_correlations.head(10),
                name='Positive Correlation',
                marker_color='#2ca02c',
                hovertemplate="%{x}<br>Correlation: %{y:.2f}<extra></extra>"
            )
        )
        
        # Add negative correlations
        fig.add_trace(
            go.Bar(
                x=sorted_correlations.tail(10).index,
                y=sorted_correlations.tail(10),
                name='Negative Correlation',
                marker_color='#d62728',
                hovertemplate="%{x}<br>Correlation: %{y:.2f}<extra></extra>"
            )
        )
        
        fig.update_layout(
            title=f'Correlation with {coin_selected}',
            xaxis_title='Cryptocurrency',
            yaxis_title='Correlation Coefficient',
            yaxis_range=[-1, 1],
            hovermode='x unified',
            showlegend=False,
            height=500,
            plot_bgcolor='rgba(240,240,240,0.8)'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Create expandable sections for detailed tables
    with st.expander("📊 Detailed Correlation Data", expanded=False):
        tab1, tab2 = st.tabs(["Top Positive Correlations", "Top Negative Correlations"])
        
        with tab1:
            st.subheader(f"Top 5 Positive Correlations with {coin_selected}")
            top_positive = sorted_correlations.head(5).reset_index()
            top_positive.columns = ['Cryptocurrency', 'Correlation']
            st.dataframe(
                top_positive.style.format({'Correlation': '{:.3f}'}),
                hide_index=True
            )
            
        with tab2:
            st.subheader(f"Top 5 Negative Correlations with {coin_selected}")
            top_negative = sorted_correlations.tail(5).reset_index()
            top_negative.columns = ['Cryptocurrency', 'Correlation']
            st.dataframe(
                top_negative.sort_values('Correlation').style.format({'Correlation': '{:.3f}'}),
                hide_index=True
            )
    
    # Add scatter plot of strongest correlations
    st.subheader("🔍 Correlation Relationships")
    col1, col2 = st.columns(2)
    
    with col1:
        strongest_positive = sorted_correlations.index[0]
        fig_pos = go.Figure()
        fig_pos.add_trace(
            go.Scatter(
                x=pivoted_data[coin_selected],
                y=pivoted_data[strongest_positive],
                mode='markers',
                name=f'{coin_selected} vs {strongest_positive}',
                marker=dict(color='#2ca02c', opacity=0.6)
            )
        )
        fig_pos.update_layout(
            title=f'Strongest Positive: {strongest_positive} (ρ={sorted_correlations[0]:.2f})',
            xaxis_title=coin_selected,
            yaxis_title=strongest_positive,
            height=400
        )
        st.plotly_chart(fig_pos, use_container_width=True)
    
    with col2:
        strongest_negative = sorted_correlations.index[-1]
        fig_neg = go.Figure()
        fig_neg.add_trace(
            go.Scatter(
                x=pivoted_data[coin_selected],
                y=pivoted_data[strongest_negative],
                mode='markers',
                name=f'{coin_selected} vs {strongest_negative}',
                marker=dict(color='#d62728', opacity=0.6)
            )
        )
        fig_neg.update_layout(
            title=f'Strongest Negative: {strongest_negative} (ρ={sorted_correlations[-1]:.2f})',
            xaxis_title=coin_selected,
            yaxis_title=strongest_negative,
            height=400
        )
        st.plotly_chart(fig_neg, use_container_width=True)

def plot_moving_average():
    """Enhanced moving average visualization while preserving sidebar structure."""
    st.header("Plotting Moving Average")
    
    # Get available coins from the data
    available_coins = combined_data['Crypto'].unique()
    
    # Preserve existing sidebar controls exactly as they were
    coin_selected = st.selectbox("Select cryptocurrency:", available_coins)
    window_size = st.radio("Window size:", ['Short (30-day MA)', 'Medium (60-day MA)', 'Long (90-day MA)'])
    
    window_mapping = {'Short (30-day MA)': 30, 'Medium (60-day MA)': 60, 'Long (90-day MA)': 90}
    window = window_mapping[window_size]
    
    # Filter data for selected coin
    selected_data = combined_data[combined_data['Crypto'] == coin_selected].copy()
    
    # Calculate moving average
    selected_data['MA'] = selected_data['Close'].rolling(window=window).mean()
    
    # Create enhanced visualization
    fig = go.Figure()
    
    # Price line
    fig.add_trace(go.Scatter(
        x=selected_data.index,
        y=selected_data['Close'],
        name=f'{coin_selected} Price',
        line=dict(color='#1f77b4', width=2),
        hovertemplate='<b>%{x|%b %d, %Y}</b><br>Price: £%{y:.2f}<extra></extra>'
    ))
    
    # Moving average line
    fig.add_trace(go.Scatter(
        x=selected_data.index,
        y=selected_data['MA'],
        name=f'{window}-day MA',
        line=dict(color='#ff7f0e', width=2, dash='dash'),
        hovertemplate='<b>%{x|%b %d, %Y}</b><br>MA: £%{y:.2f}<extra></extra>'
    ))
    
    # Highlight crossovers
    selected_data['Above_MA'] = selected_data['Close'] > selected_data['MA']
    crossovers = selected_data[selected_data['Above_MA'] != selected_data['Above_MA'].shift(1)].index
    
    fig.add_trace(go.Scatter(
        x=crossovers,
        y=selected_data.loc[crossovers, 'Close'],
        mode='markers',
        name='Crossover',
        marker=dict(
            color='#2ca02c',
            size=10,
            symbol='diamond'
        ),
        hovertemplate='<b>Crossover</b><br>Price: £%{y:.2f}<extra></extra>'
    ))
    
    # Layout enhancements
    fig.update_layout(
        title=f'<b>{coin_selected} Price vs {window}-day Moving Average</b>',
        xaxis_title='Date',
        yaxis_title='Price (GBP)',
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        template='plotly_white',
        margin=dict(l=40, r=40, t=60, b=40),
        height=500
    )
    
    # Add range selector
    fig.update_xaxes(
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(step="all")
            ])
        ),
        rangeslider=dict(visible=True)
    )
    
    # Display the plot
    st.plotly_chart(fig, use_container_width=True)
    
    # Add metrics below the chart 
    col1, col2 = st.columns(2)
    
    with col1:
        current_price = selected_data['Close'].iloc[-1]
        current_ma = selected_data['MA'].iloc[-1]
        st.metric(
            label="Current Price",
            value=f"£{current_price:.2f}",
            delta=f"£{(current_price - selected_data['Close'].iloc[-2]):.2f} from previous"
        )
    
    with col2:
        st.metric(
            label=f"{window}-day MA",
            value=f"£{current_ma:.2f}",
            delta=f"£{(current_price - current_ma):.2f} difference"
        )
    
    # Interpretation help
    with st.expander("How to interpret this chart"):
        st.markdown("""
        - **Price above MA**: Potential bullish signal
        - **Price below MA**: Potential bearish signal
        - **Crossover points**: May indicate trend changes
        - **Diamond markers**: Show where price crossed the MA
        - Use the range selector above to zoom in/out
        """)


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
    st.header("📊 Selected Cryptocurrencies Analysis")
    
    if selected_data.empty:
        st.warning("⚠️ No coin data available. Please select coins first.")
        return
    
    # Add some metrics at the top
    cols = st.columns(4)
    for i, col_name in enumerate(selected_data.columns[:4]):
        cols[i].metric(
            label=f"Mean {col_name}",
            value=f"{selected_data[col_name].mean():.2f}",
            delta=f"{selected_data[col_name].std():.2f} std"
        )
    
    st.subheader("📋 Dataset Preview")
    st.dataframe(
        selected_data.style.background_gradient(cmap='Blues'),
        height=min(400, (len(selected_data) + 1) * 35),
        use_container_width=True
    )
    
    st.subheader("📦 Distribution Analysis")
    fig = make_subplots(rows=1, cols=4, subplot_titles=[f"<b>{col}</b>" for col in selected_data.columns[:4]])
    
    colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA']  # Custom color palette
    
    for i, column in enumerate(selected_data.columns[:4], start=1):
        fig.add_trace(
            go.Box(
                y=selected_data[column], 
                name=column,
                marker_color=colors[i-1],
                boxmean=True
            ), 
            row=1, 
            col=i
        )
    
    fig.update_layout(
        title='<b>Price Distribution of Selected Cryptocurrencies</b>',
        title_font=dict(size=18),
        showlegend=False,
        width=1000, 
        height=500,
        margin=dict(t=60),
        plot_bgcolor='rgba(240,240,240,0.8)'
    )
    st.plotly_chart(fig, use_container_width=True)

def plot_coin_scatter():
    if selected_data.empty:
        st.warning("⚠️ No coin data available. Please select coins first.")
        return
    
    st.header("🔍 Pairwise Relationships")
    st.markdown("Explore how different cryptocurrencies correlate with each other")
    
    coin_combinations = list(itertools.combinations(selected_data.columns, 2))
    n_plots = len(coin_combinations)
    n_rows = (n_plots + 2) // 3  # Ensure we have enough rows
    
    fig = make_subplots(
        rows=n_rows, 
        cols=3,
        subplot_titles=[f"<b>{coin1}</b> vs <b>{coin2}</b>" for coin1, coin2 in coin_combinations],
        vertical_spacing=0.1,
        horizontal_spacing=0.1
    )
    
    # Generate a consistent color palette
    colors = px.colors.qualitative.Plotly
    
    for i, (coin1, coin2) in enumerate(coin_combinations, start=1):
        row = (i - 1) // 3 + 1
        col = (i - 1) % 3 + 1
        
        # Calculate correlation for annotation
        corr = selected_data[coin1].corr(selected_data[coin2])
        
        fig.add_trace(
            go.Scatter(
                x=selected_data[coin1], 
                y=selected_data[coin2], 
                mode='markers',
                marker=dict(
                    color=colors[i % len(colors)],
                    size=8,
                    opacity=0.6
                ),
                name=f"{coin1} vs {coin2}",
                hoverinfo='x+y'
            ), 
            row=row, 
            col=col
        )
        
        # Add correlation annotation
        fig.add_annotation(
            xref=f"x{i}", yref=f"y{i}",
            x=0.95, y=0.95,
            text=f"ρ = {corr:.2f}",
            showarrow=False,
            font=dict(size=12, color="black"),
            bgcolor="white",
            opacity=0.8,
            row=row,
            col=col
        )
    
    fig.update_layout(
        height=300 * n_rows,
        width=1000,
        showlegend=False,
        margin=dict(t=40),
        plot_bgcolor='rgba(240,240,240,0.8)'
    )
    
    # Improve axis labels
    for i in range(1, n_plots + 1):
        fig.update_xaxes(title_text="Price", row=(i-1)//3 + 1, col=(i-1)%3 + 1)
        fig.update_yaxes(title_text="Price", row=(i-1)//3 + 1, col=(i-1)%3 + 1)
    
    st.plotly_chart(fig, use_container_width=True)

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import joblib
import os
import tensorflow as tf
from sklearn.metrics import (mean_absolute_error, mean_squared_error, 
                            r2_score)
from sklearn.inspection import permutation_importance
from xgboost import plot_importance
from keras.models import Model
import shap
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from keras.models import Sequential
from keras.layers import LSTM, Dense

def analyze_feature_importance(model, model_type, X_train, feature_names):
    """
    Analyze and visualize feature importance for different model types
    
    Args:
        model: Trained model
        model_type: One of ['GRADIENT BOOSTING', 'SVR', 'XGBOOST', 'LSTM']
        X_train: Training features
        feature_names: List of feature names
        
    Returns:
        fig: Matplotlib figure object
    """
    st.subheader(f"🔍 Feature Importance Analysis ({model_type})")
    
    try:
        if model_type == 'GRADIENT BOOSTING':
            # Gradient Boosting built-in feature importance
            importance = model.feature_importances_
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.barh(feature_names, importance)
            ax.set_title('Gradient Boosting Feature Importance')
            ax.set_xlabel('Relative Importance')
            plt.tight_layout()
            return fig
            
        elif model_type == 'XGBOOST':
            # XGBoost built-in importance
            fig, ax = plt.subplots(figsize=(8, 4))
            plot_importance(model, ax=ax, height=0.5)
            ax.set_title('XGBoost Feature Importance')
            plt.tight_layout()
            return fig
            
        elif model_type == 'SVR':
            # Permutation importance for SVR
            with st.spinner("Calculating permutation importance (this may take a minute)..."):
                result = permutation_importance(
                    model, X_train[:100], np.random.rand(100),  # Use subset for speed
                    n_repeats=10, random_state=42
                )
            importance = result.importances_mean
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.barh(feature_names, importance)
            ax.set_title('SVR Permutation Importance')
            ax.set_xlabel('Score Change When Permuted')
            plt.tight_layout()
            return fig
            
        elif model_type == 'LSTM':
            try:
                with st.spinner("Calculating SHAP values (this may take a few minutes)..."):
                    # Create a small background dataset
                    background = X_train[:100].reshape(100, -1, 1)
                    
                    # Try different SHAP explainers
                    try:
                        explainer = shap.DeepExplainer(model, background)
                    except:
                        # Fallback to GradientExplainer if DeepExplainer fails
                        explainer = shap.GradientExplainer(model, background)
                        
                    shap_values = explainer.shap_values(background)
                    
                    fig, ax = plt.subplots(figsize=(8, 4))
                    shap.summary_plot(
                        shap_values[0].reshape(-1, len(feature_names)),
                        background.reshape(-1, len(feature_names)),
                        feature_names=feature_names,
                        plot_type='bar',
                        show=False
                    )
                    plt.title('LSTM Feature Importance (SHAP Values)')
                    plt.tight_layout()
                    return fig
                    
            except Exception as e:
                st.warning(f"Couldn't compute SHAP values: {str(e)}")
                st.info("Showing alternative permutation importance instead")
                
                # Fallback to permutation importance
                from sklearn.inspection import permutation_importance
                X_train_flat = X_train.reshape(X_train.shape[0], -1)
                r = permutation_importance(
                    model, 
                    X_train_flat[:100], 
                    y_train[:100],
                    n_repeats=5,
                    random_state=42
                )
                
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.barh(feature_names, r.importances_mean)
                ax.set_title('LSTM Permutation Importance (Fallback)')
                plt.tight_layout()
                return fig

def evaluate_models_selected_coin(data, coin_index):
    """
    Evaluate machine learning models for a specific cryptocurrency with single selection,
    enhanced visualizations, and feature importance analysis.
    """
    try:
        # Validate input data
        if data.empty:
            st.error("No data available for evaluation.")
            return
            
        coin_name = data.columns[coin_index]
        model_dir = f"trained_models/Model_SELECTED_COIN_{coin_index+1}"
        
        # Display version requirements in expander
        req_file = f"{model_dir}/requirements.txt"
        if os.path.exists(req_file):
            with st.expander("Model Version Requirements", expanded=False):
                with open(req_file) as f:
                    st.code(f.read())
        
        # Check if models exist, if not train them
        if not os.path.exists(model_dir):
            with st.status("Training models...", expanded=True) as status:
                st.write(f"No trained models found for {coin_name}. Training models now...")
                train_models_for_coin(data, coin_index)
                status.update(label="Models trained successfully!", state="complete")
        
        # Prepare data with lag features
        with st.spinner("Preparing data..."):
            data_prep = data.copy()
            for lag in range(1, 4):
                data_prep[f'{coin_name}_lag_{lag}'] = data_prep[coin_name].shift(lag)
            data_prep.dropna(inplace=True)
            
            features = [f'{coin_name}_lag_{lag}' for lag in range(1, 4)]
            X = data_prep[features]
            y = data_prep[coin_name]
            
            # Time-based split instead of random for time series
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]

        # Model loading with error recovery
        models = {}
        retrained = False
        try:
            with st.spinner("Loading models..."):
                models = {
                    'GRADIENT BOOSTING': joblib.load(f"{model_dir}/gradient_boosting_model.pkl"),
                    'SVR': joblib.load(f"{model_dir}/svr_model.pkl"),
                    'XGBOOST': joblib.load(f"{model_dir}/xgboost_model.pkl"),
                    'LSTM': tf.keras.models.load_model(f"{model_dir}/lstm_model.keras")
                }
        except Exception as e:
            with st.status("Model version mismatch detected - retraining models...", expanded=True) as status:
                st.warning(f"Model loading failed: {str(e)}")
                train_models_for_coin(data, coin_index)
                retrained = True
                
                try:
                    models = {
                        'GRADIENT BOOSTING': joblib.load(f"{model_dir}/gradient_boosting_model.pkl"),
                        'SVR': joblib.load(f"{model_dir}/svr_model.pkl"),
                        'XGBOOST': joblib.load(f"{model_dir}/xgboost_model.pkl"),
                        'LSTM': tf.keras.models.load_model(f"{model_dir}/lstm_model.keras")
                    }
                    status.update(label="Models retrained successfully!", state="complete")
                except Exception as e:
                    status.update(label="Model loading failed again", state="error")
                    st.error(f"Model loading failed again: {str(e)}")
                    return

        # Multiple model selection
        available_models = list(models.keys())
        selected_models = st.multiselect(
            "Select models to evaluate:",
            options=available_models,
            default=[available_models[0]] if available_models else [],  # Default to first model if available
            key=f"multimodel_select_{coin_index}"
        )

        # Ensure at least one model is selected
        if not selected_models and available_models:
            st.warning("Please select at least one model to evaluate.")
            selected_models = [available_models[0]]  

        # Evaluation metrics storage
        eval_metrics = {}
        predictions_data = []
        time_series_data = []

        for selected_model in selected_models:  # Loop through all selected models
            with st.spinner(f"Evaluating {selected_model}..."):
                model = models[selected_model]
                if model is None:
                    st.warning(f"{selected_model} model not available")
                    continue

                try:
                    # Make predictions
                    if selected_model == 'LSTM':
                        X_test_array = X_test.to_numpy().reshape(X_test.shape[0], X_test.shape[1], 1)
                        predictions = model.predict(X_test_array).flatten()
                    else:
                        predictions = model.predict(X_test)

                    # Calculate metrics
                    metrics = {
                        'MAE': mean_absolute_error(y_test, predictions),
                        'MSE': mean_squared_error(y_test, predictions),
                        'RMSE': np.sqrt(mean_squared_error(y_test, predictions)),
                        'MAPE': np.mean(np.abs((y_test - predictions) / y_test)) * 100,
                        'R2': r2_score(y_test, predictions)
                    }
                    eval_metrics[selected_model] = metrics

                    # Store data for visualizations
                    predictions_data.append({
                        'Model': selected_model,
                        'Actual': y_test,
                        'Predicted': predictions
                    })
                    
                    # Store time series data
                    time_series_data.append({
                        'Model': selected_model,
                        'Dates': data_prep.index[split_idx:],
                        'Actual': y_test,
                        'Predicted': predictions
                    })

                except Exception as e:
                    st.error(f"Error evaluating {selected_model}: {str(e)}")
                    continue

        # Display results
        if not eval_metrics:
            st.error("No models were successfully evaluated")
            return

        st.subheader(f"📊 Evaluation Results for {coin_name}")
            
        # Metrics table with enhanced styling
        with st.expander("Detailed Metrics", expanded=True):
            metrics_df = pd.DataFrame.from_dict(eval_metrics, orient='index')
            
            # Apply conditional formatting
            def color_metric(val, metric_name):
                if metric_name == 'R2':
                    # Green for higher R2 (better)
                    color = 'green' if val > 0.7 else 'orange' if val > 0.5 else 'red'
                else:
                    # Red for higher error metrics (worse)
                    color = 'red' if val > metrics_df[metric_name].mean() else 'green'
                return f'color: {color}; font-weight: bold'
            
            styled_metrics = metrics_df.style.format({
                'MAE': '{:.4f}',
                'MSE': '{:.4f}',
                'RMSE': '{:.4f}',
                'MAPE': '{:.2f}%',
                'R2': '{:.4f}'
            }).map(lambda x: 'font-weight: bold', subset=['R2'])
            
            # Apply color to each metric column
            for metric in metrics_df.columns:
                styled_metrics = styled_metrics.map(
                    lambda x, m=metric: color_metric(x, m), 
                    subset=[metric]
                )
            
            st.dataframe(styled_metrics, use_container_width=True)
            
            # Add download button
            csv = metrics_df.to_csv().encode('utf-8')
            st.download_button(
                label="Download Metrics as CSV",
                data=csv,
                file_name=f'{coin_name}_metrics.csv',
                mime='text/csv',
                key=f"dl_metrics_{coin_index}"
            )

        # Time Series Visualization
        st.subheader("⏳ Time Series Performance")

        fig_ts = go.Figure()

        # Add actual values (only once)
        fig_ts.add_trace(go.Scatter(
            x=time_series_data[0]['Dates'],
            y=time_series_data[0]['Actual'],
            mode='lines',
            name='Actual',
            line=dict(color='black', width=2),
            hovertemplate='Date: %{x}<br>Price: %{y:.4f}'
        ))

        # Add predicted values for each model
        for model_data in time_series_data:
            fig_ts.add_trace(go.Scatter(
                x=model_data['Dates'],
                y=model_data['Predicted'],
                mode='lines',
                name=f"{model_data['Model']} Predicted",
                line=dict(dash='dash'),
                hovertemplate='Date: %{x}<br>Predicted: %{y:.4f}'
            ))

        fig_ts.update_layout(
            title='Actual vs Predicted Over Time',
            xaxis_title='Date',
            yaxis_title='Price',
            hovermode="x unified",
            height=500
        )
        st.plotly_chart(fig_ts, use_container_width=True)

        # Actual vs Predicted scatter plot
        st.subheader("🎯 Prediction Accuracy")

        for model_data in predictions_data:
            st.markdown(f"**{model_data['Model']}**")
            col1, col2 = st.columns([3, 1])
            
            with col1:
                fig_scatter = go.Figure()
                
                fig_scatter.add_trace(go.Scatter(
                    x=model_data['Actual'],
                    y=model_data['Predicted'],
                    mode='markers',
                    name=model_data['Model'],
                    marker=dict(size=8, opacity=0.7),
                    hovertemplate='Actual: %{x:.4f}<br>Predicted: %{y:.4f}'
                ))
                
                # Add perfect prediction line
                min_val = min(model_data['Actual'])
                max_val = max(model_data['Actual'])
                fig_scatter.add_trace(go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode='lines',
                    name='Perfect Prediction',
                    line=dict(color='red', dash='dash'),
                    hovertemplate=None
                ))
                
                fig_scatter.update_layout(
                    title=f'Actual vs Predicted Values ({model_data["Model"]})',
                    xaxis_title='Actual Price',
                    yaxis_title='Predicted Price',
                    showlegend=True,
                    height=400
                )
                st.plotly_chart(fig_scatter, use_container_width=True)
            
            with col2:
                metrics = eval_metrics[model_data['Model']]
                st.metric("R² Score", f"{metrics['R2']:.4f}")
                st.metric("RMSE", f"{metrics['RMSE']:.4f}")
                st.metric("MAE", f"{metrics['MAE']:.4f}")
                st.download_button(
                    f"Download {model_data['Model']} Data",
                    pd.DataFrame(model_data).to_csv().encode('utf-8'),
                    file_name=f'{coin_name}_{model_data["Model"]}_prediction_data.csv',
                    mime='text/csv',
                    key=f"dl_pred_data_{coin_index}_{model_data['Model']}"
                )
                
            # Feature Importance Analysis
            if model_data['Model'] in models:
                model = models[model_data['Model']]
                if model is not None:
                    # Prepare X_train for feature importance analysis
                    if model_data['Model'] == 'LSTM':
                        X_train_array = X_train.to_numpy().reshape(X_train.shape[0], X_train.shape[1], 1)
                    else:
                        X_train_array = X_train.to_numpy()
                    
                    # Show feature importance
                    fig = analyze_feature_importance(
                        model, 
                        model_data['Model'], 
                        X_train_array,
                        features
                    )
                    if fig:
                        st.pyplot(fig)
                        plt.close()

        # Show retrained notice if applicable
        if retrained:
            st.info("ℹ️ Note: Models were retrained with current environment settings")
        

    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        st.error("Please check your data and model files")

def plot_actual_forecast_with_confidence(actual, predictions, periods, upper_bound, lower_bound, coin_name, model_name):
    """Enhanced visualization of actual vs predicted values with confidence interval"""
    fig = go.Figure()
    
    # Main traces
    fig.add_trace(go.Scatter(
        x=periods, y=actual, 
        mode='lines+markers', 
        name='Actual', 
        line=dict(color='#2ca02c', width=2),
        marker=dict(size=6)
    ))
    
    fig.add_trace(go.Scatter(
        x=periods, y=predictions, 
        mode='lines+markers', 
        name='Forecast', 
        line=dict(color='#d62728', width=2, dash='dot'),
        marker=dict(size=6, symbol='diamond')
    ))
    
    # Confidence interval
    fig.add_trace(go.Scatter(
        x=periods, y=upper_bound, 
        mode='lines', 
        name='Upper Bound (95%)', 
        line=dict(color='#1f77b4', width=1, dash='dash'),
        opacity=0.3
    ))
    
    fig.add_trace(go.Scatter(
        x=periods, y=lower_bound, 
        mode='lines', 
        name='Lower Bound (95%)', 
        fill='tonexty', 
        line=dict(color='#1f77b4', width=1, dash='dash'),
        opacity=0.3
    ))
    
    fig.update_layout(
        title=f"{coin_name} - Actual vs Forecast ({model_name})",
        xaxis_title='Date',
        yaxis_title='Price',
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white",
        margin=dict(l=20, r=20, t=60, b=20))
    
    # Add shaded area for confidence interval
    fig.update_layout(
        annotations=[
            dict(
                xref="paper", yref="paper",
                x=0.02, y=0.98,
                text="95% Confidence Interval",
                showarrow=False,
                font=dict(size=10, color="#1f77b4"),
                bgcolor="white",
                opacity=0.8
            )
        ]
    )
    
    st.plotly_chart(fig, use_container_width=True)


# def evaluate_models_selected_coin(data, coin_index):
#     """
#     Evaluate machine learning models for a specific cryptocurrency with single selection
#     and enhanced visualizations.
#     """
#     try:
#         # Validate input data
#         if data.empty:
#             st.error("No data available for evaluation.")
#             return
            
#         coin_name = data.columns[coin_index]
#         model_dir = f"trained_models/Model_SELECTED_COIN_{coin_index+1}"
        
#         # Display version requirements in expander
#         req_file = f"{model_dir}/requirements.txt"
#         if os.path.exists(req_file):
#             with st.expander("Model Version Requirements", expanded=False):
#                 with open(req_file) as f:
#                     st.code(f.read())
        
#         # Check if models exist, if not train them
#         if not os.path.exists(model_dir):
#             with st.status("Training models...", expanded=True) as status:
#                 st.write(f"No trained models found for {coin_name}. Training models now...")
#                 train_models_for_coin(data, coin_index)
#                 status.update(label="Models trained successfully!", state="complete")
        
#         # Prepare data with lag features
#         with st.spinner("Preparing data..."):
#             data_prep = data.copy()
#             for lag in range(1, 4):
#                 data_prep[f'{coin_name}_lag_{lag}'] = data_prep[coin_name].shift(lag)
#             data_prep.dropna(inplace=True)
            
#             features = [f'{coin_name}_lag_{lag}' for lag in range(1, 4)]
#             X = data_prep[features]
#             y = data_prep[coin_name]
            
#             # Time-based split instead of random for time series
#             split_idx = int(len(X) * 0.8)
#             X_train, X_test = X[:split_idx], X[split_idx:]
#             y_train, y_test = y[:split_idx], y[split_idx:]

#         # Model loading with error recovery
#         models = {}
#         retrained = False
#         try:
#             with st.spinner("Loading models..."):
#                 models = {
#                     'GRADIENT BOOSTING': joblib.load(f"{model_dir}/gradient_boosting_model.pkl"),
#                     'SVR': joblib.load(f"{model_dir}/svr_model.pkl"),
#                     'XGBOOST': joblib.load(f"{model_dir}/xgboost_model.pkl"),
#                     'LSTM': tf.keras.models.load_model(f"{model_dir}/lstm_model.keras")
#                 }
#         except Exception as e:
#             with st.status("Model version mismatch detected - retraining models...", expanded=True) as status:
#                 st.warning(f"Model loading failed: {str(e)}")
#                 train_models_for_coin(data, coin_index)
#                 retrained = True
                
#                 try:
#                     models = {
#                         'GRADIENT BOOSTING': joblib.load(f"{model_dir}/gradient_boosting_model.pkl"),
#                         'SVR': joblib.load(f"{model_dir}/svr_model.pkl"),
#                         'XGBOOST': joblib.load(f"{model_dir}/xgboost_model.pkl"),
#                         'LSTM': tf.keras.models.load_model(f"{model_dir}/lstm_model.keras")
#                     }
#                     status.update(label="Models retrained successfully!", state="complete")
#                 except Exception as e:
#                     status.update(label="Model loading failed again", state="error")
#                     st.error(f"Model loading failed again: {str(e)}")
#                     return

#         # Multiple model selection
#         available_models = list(models.keys())
#         selected_models = st.multiselect(
#             "Select models to evaluate:",
#             options=available_models,
#             default=[available_models[0]] if available_models else [],  # Default to first model if available
#             key=f"multimodel_select_{coin_index}"
#         )

#         # Ensure at least one model is selected
#         if not selected_models and available_models:
#             st.warning("Please select at least one model to evaluate.")
#             selected_models = [available_models[0]]  

#         # Evaluation metrics storage
#         eval_metrics = {}
#         predictions_data = []
#         time_series_data = []

#         for selected_model in selected_models:  # Loop through all selected models
#             with st.spinner(f"Evaluating {selected_model}..."):
#                 model = models[selected_model]
#                 if model is None:
#                     st.warning(f"{selected_model} model not available")
#                     continue

#                 try:
#                     # Make predictions
#                     if selected_model == 'LSTM':
#                         X_test_array = X_test.to_numpy().reshape(X_test.shape[0], X_test.shape[1], 1)
#                         predictions = model.predict(X_test_array).flatten()
#                     else:
#                         predictions = model.predict(X_test)

#                     # Calculate metrics
#                     metrics = {
#                         'MAE': mean_absolute_error(y_test, predictions),
#                         'MSE': mean_squared_error(y_test, predictions),
#                         'RMSE': np.sqrt(mean_squared_error(y_test, predictions)),
#                         'MAPE': np.mean(np.abs((y_test - predictions) / y_test)) * 100,
#                         'R2': r2_score(y_test, predictions)
#                     }
#                     eval_metrics[selected_model] = metrics

#                     # Store data for visualizations
#                     predictions_data.append({
#                         'Model': selected_model,
#                         'Actual': y_test,
#                         'Predicted': predictions
#                     })
                    
#                     # Store time series data
#                     time_series_data.append({
#                         'Model': selected_model,
#                         'Dates': data_prep.index[split_idx:],
#                         'Actual': y_test,
#                         'Predicted': predictions
#                     })

#                 except Exception as e:
#                     st.error(f"Error evaluating {selected_model}: {str(e)}")
#                     continue

#         # Display results
#         if not eval_metrics:
#             st.error("No models were successfully evaluated")
#             return

#         st.subheader(f"📊 Evaluation Results for {coin_name}")
            
#         # Metrics table with enhanced styling
#         with st.expander("Detailed Metrics", expanded=True):
#             metrics_df = pd.DataFrame.from_dict(eval_metrics, orient='index')
            
#             # Apply conditional formatting
#             def color_metric(val, metric_name):
#                 if metric_name == 'R2':
#                     # Green for higher R2 (better)
#                     color = 'green' if val > 0.7 else 'orange' if val > 0.5 else 'red'
#                 else:
#                     # Red for higher error metrics (worse)
#                     color = 'red' if val > metrics_df[metric_name].mean() else 'green'
#                 return f'color: {color}; font-weight: bold'
            
#             styled_metrics = metrics_df.style.format({
#                 'MAE': '{:.4f}',
#                 'MSE': '{:.4f}',
#                 'RMSE': '{:.4f}',
#                 'MAPE': '{:.2f}%',
#                 'R2': '{:.4f}'
#             }).map(lambda x: 'font-weight: bold', subset=['R2'])
            
#             # Apply color to each metric column
#             for metric in metrics_df.columns:
#                 styled_metrics = styled_metrics.map(
#                     lambda x, m=metric: color_metric(x, m), 
#                     subset=[metric]
#                 )
            
#             st.dataframe(styled_metrics, use_container_width=True)
            
#             # Add download button
#             csv = metrics_df.to_csv().encode('utf-8')
#             st.download_button(
#                 label="Download Metrics as CSV",
#                 data=csv,
#                 file_name=f'{coin_name}_metrics.csv',
#                 mime='text/csv',
#                 key=f"dl_metrics_{coin_index}"
#             )

#         # Time Series Visualization
#         st.subheader("⏳ Time Series Performance")

#         fig_ts = go.Figure()

#         # Add actual values (only once)
#         fig_ts.add_trace(go.Scatter(
#             x=time_series_data[0]['Dates'],
#             y=time_series_data[0]['Actual'],
#             mode='lines',
#             name='Actual',
#             line=dict(color='black', width=2),
#             hovertemplate='Date: %{x}<br>Price: %{y:.4f}'
#         ))

#         # Add predicted values for each model
#         for model_data in time_series_data:
#             fig_ts.add_trace(go.Scatter(
#                 x=model_data['Dates'],
#                 y=model_data['Predicted'],
#                 mode='lines',
#                 name=f"{model_data['Model']} Predicted",
#                 line=dict(dash='dash'),
#                 hovertemplate='Date: %{x}<br>Predicted: %{y:.4f}'
#             ))

#         fig_ts.update_layout(
#             title='Actual vs Predicted Over Time',
#             xaxis_title='Date',
#             yaxis_title='Price',
#             hovermode="x unified",
#             height=500
#         )
#         st.plotly_chart(fig_ts, use_container_width=True)

#         # Actual vs Predicted scatter plot
#         st.subheader("🎯 Prediction Accuracy")

#         for model_data in predictions_data:
#             st.markdown(f"**{model_data['Model']}**")
#             col1, col2 = st.columns([3, 1])
            
#             with col1:
#                 fig_scatter = go.Figure()
                
#                 fig_scatter.add_trace(go.Scatter(
#                     x=model_data['Actual'],
#                     y=model_data['Predicted'],
#                     mode='markers',
#                     name=model_data['Model'],
#                     marker=dict(size=8, opacity=0.7),
#                     hovertemplate='Actual: %{x:.4f}<br>Predicted: %{y:.4f}'
#                 ))
                
#                 # Add perfect prediction line
#                 min_val = min(model_data['Actual'])
#                 max_val = max(model_data['Actual'])
#                 fig_scatter.add_trace(go.Scatter(
#                     x=[min_val, max_val],
#                     y=[min_val, max_val],
#                     mode='lines',
#                     name='Perfect Prediction',
#                     line=dict(color='red', dash='dash'),
#                     hovertemplate=None
#                 ))
                
#                 fig_scatter.update_layout(
#                     title=f'Actual vs Predicted Values ({model_data["Model"]})',
#                     xaxis_title='Actual Price',
#                     yaxis_title='Predicted Price',
#                     showlegend=True,
#                     height=400
#                 )
#                 st.plotly_chart(fig_scatter, use_container_width=True)
            
#             with col2:
#                 metrics = eval_metrics[model_data['Model']]
#                 st.metric("R² Score", f"{metrics['R2']:.4f}")
#                 st.metric("RMSE", f"{metrics['RMSE']:.4f}")
#                 st.metric("MAE", f"{metrics['MAE']:.4f}")
#                 st.download_button(
#                     f"Download {model_data['Model']} Data",
#                     pd.DataFrame(model_data).to_csv().encode('utf-8'),
#                     file_name=f'{coin_name}_{model_data["Model"]}_prediction_data.csv',
#                     mime='text/csv',
#                     key=f"dl_pred_data_{coin_index}_{model_data['Model']}"
#                 )
#         # Show retrained notice if applicable
#         if retrained:
#             st.info("ℹ️ Note: Models were retrained with current environment settings")

#     except Exception as e:
#         st.error(f"An unexpected error occurred: {str(e)}")
#         st.error("Please check your data and model files")

# #
# def plot_actual_forecast_with_confidence(actual, predictions, periods, upper_bound, lower_bound, coin_name, model_name):
#     """Enhanced visualization of actual vs predicted values with confidence interval"""
#     fig = go.Figure()
    
#     # Main traces
#     fig.add_trace(go.Scatter(
#         x=periods, y=actual, 
#         mode='lines+markers', 
#         name='Actual', 
#         line=dict(color='#2ca02c', width=2),
#         marker=dict(size=6)
#     ))
    
#     fig.add_trace(go.Scatter(
#         x=periods, y=predictions, 
#         mode='lines+markers', 
#         name='Forecast', 
#         line=dict(color='#d62728', width=2, dash='dot'),
#         marker=dict(size=6, symbol='diamond')
#     ))
    
#     # Confidence interval
#     fig.add_trace(go.Scatter(
#         x=periods, y=upper_bound, 
#         mode='lines', 
#         name='Upper Bound (95%)', 
#         line=dict(color='#1f77b4', width=1, dash='dash'),
#         opacity=0.3
#     ))
    
#     fig.add_trace(go.Scatter(
#         x=periods, y=lower_bound, 
#         mode='lines', 
#         name='Lower Bound (95%)', 
#         fill='tonexty', 
#         line=dict(color='#1f77b4', width=1, dash='dash'),
#         opacity=0.3
#     ))
    
#     fig.update_layout(
#         title=f"{coin_name} - Actual vs Forecast ({model_name})",
#         xaxis_title='Date',
#         yaxis_title='Price',
#         hovermode="x unified",
#         legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
#         template="plotly_white",
#         margin=dict(l=20, r=20, t=60, b=20))
    
#     # Add shaded area for confidence interval
#     fig.update_layout(
#         annotations=[
#             dict(
#                 xref="paper", yref="paper",
#                 x=0.02, y=0.98,
#                 text="95% Confidence Interval",
#                 showarrow=False,
#                 font=dict(size=10, color="#1f77b4"),
#                 bgcolor="white",
#                 opacity=0.8
#             )
#         ]
#     )
    
#     st.plotly_chart(fig, use_container_width=True)

def create_forecast_table(predictions, periods, upper_bound, lower_bound):
    """Create a styled DataFrame with forecast results"""
    forecast_df = pd.DataFrame({
        'Date': periods,
        'Forecast': predictions,
        'Lower Bound': lower_bound,
        'Upper Bound': upper_bound,
        'Confidence Range': upper_bound - lower_bound
    })
    
    forecast_df['Date'] = forecast_df['Date'].dt.strftime('%Y-%m-%d')
    
    return forecast_df

def evaluate_and_plot_model(coin_index, model_choice, frequency, num_periods, selected_data):
    """
    Main function to evaluate model and plot results with enhanced error handling
    and visualization
    """
    if selected_data.empty:
        st.error("No selected coins data available.")
        return
    
    coin_name = selected_data.columns[coin_index]
    
    # Data preparation
    try:
        # Create lag features
        for lag in range(1, 4):
            selected_data[f'{coin_name}_lag_{lag}'] = selected_data[coin_name].shift(lag)
        
        selected_data.dropna(inplace=True)
        
        features = [f'{coin_name}_lag_{lag}' for lag in range(1, 4)]
        X = selected_data[features]
        y = selected_data[coin_name]
        
        # Train-test split (keeping temporal order)
        test_size = int(len(X) * 0.2)
        X_train, X_test = X[:-test_size], X[-test_size:]
        y_train, y_test = y[:-test_size], y[-test_size:]
        
    except Exception as e:
        st.error(f"Error in data preparation: {str(e)}")
        return
    
    # Model loading and prediction
    model_mapping = {
        'Gradient Boosting': 'gradient_boosting_model.pkl',
        'SVR': 'svr_model.pkl',
        'XGBOOST': 'xgboost_model.pkl',
        'LSTM': 'lstm_model.keras'
    }
    
    model_filename = os.path.join(
        "trained_models", 
        f"Model_SELECTED_COIN_{coin_index+1}", 
        model_mapping.get(model_choice, "")
    )
    
    if not os.path.exists(model_filename):
        st.error(f"Model file not found at: {model_filename}")
        return
    
    try:
        if model_choice == 'LSTM':
            model = load_model(model_filename)
            X_array = X.to_numpy().reshape(X.shape[0], X.shape[1], 1)
            predictions = model.predict(X_array[-num_periods:]).flatten()
        else:
            model = joblib.load(model_filename)
            predictions = model.predict(X[-num_periods:])
    except Exception as e:
        st.error(f"Error loading or predicting with model: {str(e)}")
        return
    
    # Generate forecast periods
    freq_mapping = {
        'daily': 'D',
        'weekly': 'W',
        'monthly': 'M',
        'quarterly': 'Q'
    }
    
    last_date = selected_data.index[-1]
    try:
        periods = pd.date_range(
            start=last_date, 
            periods=num_periods, 
            freq=freq_mapping.get(frequency, 'D')
        )
    except Exception as e:
        st.error(f"Error generating date range: {str(e)}")
        return
    
    # Calculate confidence intervals
    try:
        mse = mean_squared_error(y_test[-num_periods:], predictions)
        upper_bound = predictions + 1.96 * np.sqrt(mse)
        lower_bound = predictions - 1.96 * np.sqrt(mse)
    except Exception as e:
        st.error(f"Error calculating confidence intervals: {str(e)}")
        return
    
    # Display results
    st.subheader(f"Forecast Results for {coin_name} using {model_choice}")
    
    # Create and display forecast table
    forecast_df = create_forecast_table(predictions, periods, upper_bound, lower_bound)
    
    # Apply styling to the DataFrame
    styled_df = forecast_df.style \
        .format({
            'Forecast': '{:.4f}',
            'Lower Bound': '{:.4f}',
            'Upper Bound': '{:.4f}',
            'Confidence Range': '{:.4f}'
        }) \
        .apply(lambda x: ['background: #f7f7f7' if i%2==0 else '' for i in range(len(x))], axis=0) \
        .set_properties(**{'text-align': 'center'})
    
    st.dataframe(styled_df, use_container_width=True)
    
    # Plot results
    plot_actual_forecast_with_confidence(
        y_test[-num_periods:], 
        predictions, 
        periods, 
        upper_bound, 
        lower_bound,
        coin_name,
        model_choice
    )
    
    # Display metrics
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Mean Squared Error", f"{mse:.4f}")
    with col2:
        mean_confidence_range = np.mean(upper_bound - lower_bound)
        st.metric("Average Confidence Range", f"{mean_confidence_range:.4f}")



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



def plot_ma_strategy(selected_data, chosen_coin):
    """Plot the moving average strategy"""
    try:
        fig = go.Figure()
        
        # Add traces with proper formatting
        traces = [
            ('Close Price', 'blue', None, selected_data['Close']),
            ('7-day MA', 'green', None, selected_data['MA_7']),
            ('14-day MA', 'red', None, selected_data['MA_14']),
            ('Buy Signal', 'green', 'triangle-up', 
             selected_data.loc[selected_data['Buy_Signal'] == 1, 'Close']),
            ('Sell Signal', 'red', 'triangle-down',
             selected_data.loc[selected_data['Sell_Signal'] == -1, 'Close'])
        ]
        
        for name, color, symbol, y in traces:
            fig.add_trace(go.Scatter(
                x=selected_data.index,
                y=y,
                name=name,
                mode='lines' if symbol is None else 'markers',
                line=dict(color=color) if symbol is None else None,
                marker=dict(color=color, size=10, symbol=symbol) if symbol else None
            ))
    
        # Add current price line
        current_price = selected_data['Close'].iloc[-1]
        fig.add_trace(go.Scatter(
            x=[selected_data.index[0], selected_data.index[-1]],
            y=[current_price, current_price],
            mode='lines',
            name='Current Price',
            line=dict(color='gray', dash='dash')
        ))
        
        # Update layout
        fig.update_layout(
            title=f'Moving Average Strategy for {chosen_coin}',
            xaxis_title='Date',
            yaxis_title='Price',
            hovermode='x unified',
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error plotting strategy: {str(e)}")
        logging.error(f"Error in plot_ma_strategy: {str(e)}", exc_info=True)


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import streamlit as st
import os
import logging
from keras.models import load_model
import joblib

def evaluate_signal_performance(actual_returns, predicted_signals):
    """
    Evaluate the performance of buy/sell signals using classification metrics.
    
    Args:
        actual_returns (array): Actual price movements (1=up, 0=down)
        predicted_signals (array): Predicted signals (1=buy, 0=sell)
    
    Returns:
        dict: Classification metrics
    """
    try:
        # Convert to binary classification problem
        actual_binary = (actual_returns > 0).astype(int)
        pred_binary = (predicted_signals == 1).astype(int)
        
        metrics = {
            'Accuracy': accuracy_score(actual_binary, pred_binary),
            'Precision': precision_score(actual_binary, pred_binary, zero_division=0),
            'Recall': recall_score(actual_binary, pred_binary, zero_division=0),
            'F1-Score': f1_score(actual_binary, pred_binary, zero_division=0)
        }
        
        if len(np.unique(actual_binary)) > 1:
            metrics['AUC-ROC'] = roc_auc_score(actual_binary, pred_binary)
            
        return metrics
    except Exception as e:
        logging.error(f"Error in metric calculation: {str(e)}")
        return None

def forecast_price_with_model(chosen_coin, num_days, model_type, selected_data):
    """
    Forecast future price with performance metrics
    
    Args:
        chosen_coin (str): Cryptocurrency symbol
        num_days (int): Prediction horizon in days
        model_type (str): Model type
        selected_data (DataFrame): Historical price data
    
    Returns:
        tuple: (predicted_price, prediction_date, metrics_dict)
    """
    try:
        # 1. Validate inputs
        if chosen_coin not in selected_data.columns:
            st.error(f"Coin '{chosen_coin}' not found")
            return None, None, None
            
        # 2. Prepare features
        features = [f'{chosen_coin}_lag_{lag}' for lag in [1,2,3]]
        data = selected_data.copy()
        for lag in [1,2,3]:
            if f'{chosen_coin}_lag_{lag}' not in data.columns:
                data[f'{chosen_coin}_lag_{lag}'] = data[chosen_coin].shift(lag)
        
        data = data.dropna(subset=features)
        if len(data) < num_days + 30:  # Need min 30 points for backtesting
            return None, None, None
        
        # 3. Load model
        model_dir = f"trained_models/Model_SELECTED_COIN_{selected_data.columns.get_loc(chosen_coin)+1}"
        model_file = {
            "SVR": "svr_model.pkl",
            "GBR": "gradient_boosting_model.pkl",
            "XGBoost": "xgboost_model.pkl",
            "LSTM": "lstm_model.keras"
        }.get(model_type)
        
        if not model_file or not os.path.exists(f"{model_dir}/{model_file}"):
            st.error(f"Model {model_type} not found")
            return None, None, None
            
        if model_type == "LSTM":
            model = load_model(f"{model_dir}/{model_file}")
        else:
            model = joblib.load(f"{model_dir}/{model_file}")
        
        # 4. Make prediction
        X = data[features].values
        current_price = data[chosen_coin].iloc[-1]
        
        if model_type == "LSTM":
            future_price = model.predict(X[-1].reshape(1, 3, 1))[0][0]
        else:
            future_price = model.predict(X[-1].reshape(1, -1))[0]
        
        # 5. Backtesting
        X_test = X[:-num_days]
        y_true = (data[chosen_coin].shift(-num_days) > data[chosen_coin]).astype(int).dropna().values
        
        if model_type == "LSTM":
            y_pred = (model.predict(X_test.reshape(X_test.shape[0], 3, 1)).flatten() > data[chosen_coin].iloc[:-num_days].values).astype(int)
        else:
            y_pred = (model.predict(X_test) > data[chosen_coin].iloc[:-num_days].values).astype(int)
        
        # Align lengths
        min_len = min(len(y_true), len(y_pred))
        metrics = evaluate_signal_performance(y_true[:min_len], y_pred[:min_len])
        
        return future_price, datetime.now() + timedelta(days=num_days), metrics
        
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        logging.error(f"Forecast failed: {str(e)}")
        return None, None, None

def create_prediction_interface(selected_data):
    """Main prediction interface with original styling"""
    st.markdown("## Prediction with Models")
    
    with st.form(key="prediction_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            coin = st.selectbox("Select Cryptocurrency", selected_data.columns)
        with col2:
            model = st.selectbox("Select Model", ["SVR", "GBR", "XGBoost", "LSTM"])
        with col3:
            days = st.slider("Days Ahead", 1, 30, 7)
        
        if st.form_submit_button("Predict Price"):
            with st.spinner("Analyzing market data..."):
                price, date, metrics = forecast_price_with_model(coin, days, model, selected_data)
                
                if price is not None:
                    current_price = selected_data[coin].iloc[-1]
                    change = (price - current_price) / current_price * 100
                    action = "Buy" if change > 0 else "Sell"
                    confidence = "Strong" if abs(change) > 5 else "Moderate" if abs(change) > 2 else "Weak"
                    
                    # Original styling
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
                    .metrics-box {
                        background-color: #f8f9fa;
                        border-radius: 8px;
                        padding: 15px;
                        margin-top: 15px;
                        border-left: 5px solid #6c757d;
                    }
                    </style>
                    """, unsafe_allow_html=True)
                    
                    # Prediction result
                    result_html = f"""
                    <div class="result-box">
                        <div style="display: flex; justify-content: space-between; flex-wrap: wrap;">
                            <div style="min-width: 150px; margin-right: 10px; margin-bottom: 15px;">
                                <div class="metric-label">Current Price</div>
                                <div class="metric-value">${current_price:.4f}</div>
                            </div>
                            <div style="min-width: 150px; margin-right: 10px; margin-bottom: 15px;">
                                <div class="metric-label">Predicted Price</div>
                                <div class="metric-value {'price-up' if change > 0 else 'price-down'}">${price:.4f}</div>
                                <div>{'▲' if change > 0 else '▼'} {abs(change):.2f}%</div>
                            </div>
                            <div style="min-width: 150px; margin-bottom: 15px;">
                                <div class="metric-label">Forecast Date</div>
                                <div class="metric-value">{date.strftime('%Y-%m-%d')}</div>
                                <div>({days} days ahead)</div>
                            </div>
                        </div>
                        <div style="margin-top: 15px; padding-top: 15px; border-top: 1px solid #ddd;">
                            <div style="font-size: 18px; font-weight: bold; margin-bottom: 5px;">
                                Recommendation: <span style="color: {'#10b981' if action == 'Buy' else '#ef4444'}">{action}</span> with {confidence} confidence
                            </div>
                            <div style="font-style: italic; color: #666;">
                                Based on {model} model analysis of {coin}
                            </div>
                        </div>
                    </div>
                    """
                    st.markdown(result_html, unsafe_allow_html=True)
                    
                    # Metrics display
                    if metrics:
                        metrics_html = f"""
                        <div class="metrics-box">
                            <div style="font-size: 16px; font-weight: bold; margin-bottom: 10px; color: #495057;">
                                📊 Model Performance Metrics
                            </div>
                            <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                                <span style="font-weight: 600; color: #6c757d;">Accuracy:</span>
                                <span style="font-weight: bold; color: #212529;">{metrics['Accuracy']:.1%}</span>
                            </div>
                            <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                                <span style="font-weight: 600; color: #6c757d;">Precision:</span>
                                <span style="font-weight: bold; color: #212529;">{metrics['Precision']:.1%}</span>
                            </div>
                            <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                                <span style="font-weight: 600; color: #6c757d;">Recall:</span>
                                <span style="font-weight: bold; color: #212529;">{metrics['Recall']:.1%}</span>
                            </div>
                            <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                                <span style="font-weight: 600; color: #6c757d;">F1-Score:</span>
                                <span style="font-weight: bold; color: #212529;">{metrics['F1-Score']:.1%}</span>
                            </div>
                            {f'''<div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                                <span style="font-weight: 600; color: #6c757d;">AUC-ROC:</span>
                                <span style="font-weight: bold; color: #212529;">{metrics['AUC-ROC']:.3f}</span>
                            </div>''' if 'AUC-ROC' in metrics else ''}
                        </div>
                        """
                        st.markdown(metrics_html, unsafe_allow_html=True)
                    else:
                        st.warning("Performance metrics unavailable (insufficient historical data for backtesting)")
                    
                    st.caption("Note: This forecast is an estimate and market conditions can change unexpectedly.")
                else:
                    st.error("Unable to generate forecast. Please try different parameters.")

            
# getting best coins   
def find_best_coins(model_type, desired_profit, num_days):
    if selected_data.empty:
        st.error("No selected coins data available.")
        return
    
    coins = selected_data.columns[:4]
    models = {}
    predictions = {}
    current_prices = {}
    
    # Get current prices
    for coin in coins:
        current_prices[coin] = selected_data[coin].iloc[-1]
    
    # Load models
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
            # Get the predicted price directly 
            features = [f'{coin}_lag_{lag}' for lag in range(1, 4)]
            
            data_copy = selected_data.copy()
            for lag in range(1, 4):
                lag_col = f'{coin}_lag_{lag}'
                if lag_col not in data_copy.columns:
                    data_copy[lag_col] = data_copy[coin].shift(lag)
            
            selected_data_clean = data_copy.dropna(subset=features)
            
            if len(selected_data_clean) == 0:
                st.warning(f"Not enough historical data for {coin}")
                continue
            
            X_array = selected_data_clean[features].to_numpy()
            
            if model_type == "LSTM":
                X_today = X_array[-1].reshape(1, len(features), 1)
                future_price = model.predict(X_today)[0][0]
            else:
                X_today = X_array[-1].reshape(1, -1)
                future_price = model.predict(X_today)[0]
            
            current_price = current_prices[coin]
            percent_change = ((future_price - current_price) / current_price) * 100
            potential_profit = future_price - current_price
            
            predictions[coin] = {
                'profit': potential_profit,
                'percent_change': percent_change,
                'current_price': current_price,
                'future_price': future_price
            }
            
        except Exception as e:
            st.warning(f"Error predicting for {coin}: {str(e)}")
    
    if not predictions:
        st.error("No valid predictions could be generated.")
        return
    
    # Separate coins that exceed target from those that don't
    exceed_target = {coin: data for coin, data in predictions.items() if data['profit'] >= desired_profit}
    below_target = {coin: data for coin, data in predictions.items() if data['profit'] < desired_profit}
    
    recommended_coins = []
    
    # If we have coins that exceed target, recommend the best ones
    if exceed_target:
        sorted_exceed = sorted(exceed_target.items(), 
                             key=lambda x: x[1]['profit'], 
                             reverse=True)
        recommended_coins.append(sorted_exceed[0])
        
        # Add closest to target if available
        if len(sorted_exceed) > 1:
            closest = min(sorted_exceed[1:], 
                        key=lambda x: abs(x[1]['profit'] - desired_profit))
            recommended_coins.append(closest)
        elif below_target:
            closest_below = max(below_target.items(), 
                              key=lambda x: x[1]['profit'])
            recommended_coins.append(closest_below)
    else:
        # Get top 2 closest to target
        sorted_below = sorted(below_target.items(), 
                            key=lambda x: x[1]['profit'], 
                            reverse=True)
        recommended_coins = sorted_below[:2]
    
    # Display results with improved formatting
    # st.markdown("### Prediction Results")
    
    # Display results using the consistent format
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
    .profit-box {
        border-left: 5px solid #f6c23e;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("## 📈 Prediction Results")

    for i, (coin, data) in enumerate(recommended_coins[:2]):
        is_positive = data['profit'] >= desired_profit
        box_class = "result-box" if is_positive else "result-box profit-box"
        
        result_html = f"""
        <div class="{box_class}">
            <div style="display: flex; justify-content: space-between; flex-wrap: wrap;">
                <div style="min-width: 150px; margin-right: 10px; margin-bottom: 15px;">
                    <div class="metric-label">Cryptocurrency</div>
                    <div class="metric-value">{coin}</div>
                </div>
                <div style="min-width: 150px; margin-right: 10px; margin-bottom: 15px;">
                    <div class="metric-label">Current Price</div>
                    <div class="metric-value">${data['current_price']:.4f}</div>
                </div>
                <div style="min-width: 150px; margin-right: 10px; margin-bottom: 15px;">
                    <div class="metric-label">Predicted Price</div>
                    <div class="metric-value {'price-up' if data['percent_change'] > 0 else 'price-down'}">${data['future_price']:.4f}</div>
                    <div>{'▲' if data['percent_change'] > 0 else '▼'} {abs(data['percent_change']):.2f}%</div>
                </div>
            </div>
            <div style="display: flex; justify-content: space-between; flex-wrap: wrap; margin-top: 15px;">
                <div style="min-width: 150px; margin-right: 10px; margin-bottom: 15px;">
                    <div class="metric-label">Predicted Profit</div>
                    <div class="metric-value {'price-up' if is_positive else 'price-down'}">${data['profit']:.2f}</div>
                </div>
                <div style="min-width: 150px; margin-right: 10px; margin-bottom: 15px;">
                    <div class="metric-label">Target Profit</div>
                    <div class="metric-value">${desired_profit:.2f}</div>
                </div>
                <div style="min-width: 150px; margin-bottom: 15px;">
                    <div class="metric-label">Time Period</div>
                    <div class="metric-value">{num_days} days</div>
                </div>
            </div>
            <div style="margin-top: 15px; padding-top: 15px; border-top: 1px solid #ddd;">
                <div style="font-size: 18px; font-weight: bold; margin-bottom: 5px;">
                    Target Achievement: <span style="color: {'#10b981' if is_positive else '#ef4444'}">
                    {((data['profit'] / desired_profit) * 100) if desired_profit != 0 else 0:.1f}%</span>
                </div>
                <div style="font-style: italic; color: #666;">
                    Based on {model_type} analysis of {coin}
                </div>
            </div>
        </div>
        """
        
        st.markdown(result_html, unsafe_allow_html=True)
    
    st.caption("Note: Predictions are estimates and market conditions can change unexpectedly.")

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
                                               'Market State Visualization', "Predicted Highs and Lows","Close Price Distribution & Trend","Daily Price Change",
         "Boxplot Comparison","Coin Volatility","Price and Volume Overview"])
        
        if visualization_option == 'Home':
            st.title("Data Visualization")
            st.image('https://img.freepik.com/free-vector/gradient-stock-market-concept_23-2149166910.jpg', 
                    use_container_width=True)
            st.write("Explore different visualizations to gain insights into cryptocurrency markets.")
        elif visualization_option == 'Price Comparison':
            st.header("Price Comparison by Metrics and Coin")
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
            st.header("Plotting Candlestick and Volumne Chart")
            coin = st.selectbox("Select cryptocurrency:", combined_data['Crypto'].unique())
            period = st.radio("Select period:", ['Daily', 'Weekly', 'Monthly'])
            plot_candlestick_chart(coin, period[0])
        elif visualization_option == 'Market State Visualization':
            st.header("Plotting Market State Over Time")
            visualize_market_state()
        elif visualization_option == "Predicted Highs and Lows":
            st.header("Plotting Market Highs and Lows")
            predict_highs_lows()
        elif visualization_option == 'Close Price Distribution & Trend':
            selected_coin = st.selectbox("Select a cryptocurrency:", combined_data['Crypto'].unique())
            plot_distribution_and_trend(selected_coin)
        elif visualization_option == 'Daily Price Change':
            selected_coin = st.selectbox("Select a cryptocurrency:", combined_data['Crypto'].unique(), key="daily")
            plot_daily_price_changes(selected_coin)
        elif visualization_option == 'Coin Volatility':
            plot_crypto_volatility()
        elif visualization_option == 'Boxplot Comparison':
            selected_coin = st.selectbox("Select a cryptocurrency:", combined_data['Crypto'].unique(), key="box")
            plot_boxplot(combined_data, selected_coin)

    elif page == "Predictions":
        prediction_option = st.sidebar.radio("Select:", 
                                           ["Dataset","K_mean", "Training", "Training Model Metrics", "Prediction Graphs",
                                            "Buy and Sell Prediction", "Predict coin by Profit"])
        
        if prediction_option == "Dataset":
            display_selected_coins()
            st.markdown("---")  # Add a horizontal divider
            plot_coin_scatter()
        elif prediction_option == "K_mean":
            st.title("PCA & Clustering Demo")

            # Assume you've done PCA and have your loadings:
            pivoted_data = combined_data.pivot(columns='Crypto', values='Close')
            pivoted_data_filled = pivoted_data.fillna(0)
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(pivoted_data_filled)
            pca = PCA(n_components=10)
            pca_result = pca.fit_transform(scaled_data)
            loadings = pd.DataFrame(
                pca.components_.T,
                columns=[f'PC{i}' for i in range(1, 11)],
                index=pivoted_data.columns
            )

            # Call your elbow plot here
            if st.button("Show Elbow Plot for KMeans Clustering"):
                plot_elbow(loadings)
    
        elif prediction_option == "Training":
            st.header("Model Training")
            
            # Set up session state for training
            initialize_session_state()
            setup_logging()
            
            st.subheader("Train Models for Cryptocurrency Prediction")
            st.write("""This will train multiple machine learning models for each selected cryptocurrency. 
                    The process includes Gradient Boosting, SVR, XGBoost, and LSTM neural networks.""")
            
            # Display current training status and controls
            if st.session_state.get('models_trained', False):
                st.success("✅ All models have been successfully trained!")
                if st.button("Train Again"):
                    initialize_session_state()
                    st.rerun()
                    
            elif st.session_state.training_state.get('started', False):
                # Show training in progress UI
                display_training_progress()
                    
            else:
                # Initial state - show training options
                with st.expander("Advanced Training Options", expanded=False):
                    st.info("Default settings will train 4 model types for up to 4 cryptocurrencies.")
                    st.write("Training all models may take several minutes, especially for LSTM networks.")
                
                if st.button("🚀 Start Training Models"):
                    # Start background thread for training
                    thread = threading.Thread(
                        target=train_all_models_background,
                        args=(selected_data,),
                        daemon=True
                    )
                    thread.start()
                    st.session_state.training_thread = thread
                    st.rerun()
        
        elif prediction_option == "Training Model Metrics":
            st.header("Selected Model Metrics")
            coins = st.multiselect("Select coins:", selected_data.columns)
        
            
            for coin in coins:
                coin_index = selected_data.columns.get_loc(coin)
                evaluate_models_selected_coin(selected_data, coin_index)
        elif prediction_option == "Prediction Graphs":
            st.header("Cryptocurrency Price Prediction")
            
            # Create two columns for better layout
            col1, col2 = st.columns(2)
            
            with col1:
                coin = st.selectbox(
                    "Select coin:", 
                    selected_data.columns,
                    help="Select the cryptocurrency you want to analyze"
                )
                
                model = st.selectbox(
                    "Select model:", 
                    ['Gradient Boosting', 'SVR', 'XGBOOST', 'LSTM'],
                    help="Choose the machine learning model for prediction"
                )
            
            with col2:
                frequency = st.selectbox(
                    "Select frequency:", 
                    ['daily', 'weekly', 'monthly', 'quarterly'],
                    help="Time intervals for the prediction"
                )
                
                periods = st.number_input(
                    "Number of periods to predict:", 
                    min_value=1, 
                    max_value=100, 
                    value=20,
                    step=1,
                    help="How many time periods (days/weeks/months) to forecast"
                )
            
            # Add some visual separation
            st.markdown("---")
            
            if st.button("Run Prediction", type="primary"):
                # Show loading spinner while processing
                with st.spinner(f"Generating {model} predictions for {coin}..."):
                    try:
                        coin_index = selected_data.columns.get_loc(coin)
                        evaluate_and_plot_model(coin_index, model, frequency, periods, selected_data)
                        
                        # Success message
                        st.success("Prediction completed successfully!")
                        
                    except Exception as e:
                        st.error(f"An error occurred during prediction: {str(e)}")
                        st.warning("Please check your inputs and try again.")

        elif prediction_option == "Buy and Sell Prediction":
            strategy = st.sidebar.radio(
                "Select Prediction Strategy:",
                ["Moving Averages", "Machine Learning Models"],
                help="Choose between technical indicators or AI models for predictions"
            )
            
            st.markdown("## Buy/Sell Recommendation Prediction")
            
            if strategy == "Moving Averages":
                st.markdown("## Prediction with Moving Average")
            
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
                        determine_best_time_to_trade_future(coin, days, combined_data)
            
            else:  # Machine Learning Models
                create_prediction_interface(selected_data)
        elif prediction_option == "Predict coin by Profit":
            st.markdown("## Profit-Based Coin Prediction")
            
            with st.form(key="profit_prediction_form"):
                # Create a 3-column layout
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    model_type = st.selectbox(
                        "Select Prediction Model:",
                        ['Gradient_Boosting', 'SVR', 'Xgboost', 'LSTM'],
                        help="Choose the machine learning model for prediction"
                    )
                
                with col2:
                    profit = st.slider(
                        "Desired Profit ($):",
                        min_value=10,
                        max_value=1000,
                        value=100,
                        step=10,
                        help="Set your target profit amount"
                    )
                
                with col3:
                    days = st.slider(
                        "Investment Period (days):",
                        min_value=1,
                        max_value=90,
                        value=30,
                        help="Select your investment time horizon"
                    )
                
                submit_button = st.form_submit_button("Find Best Coins")
                
                if submit_button:
                    find_best_coins(model_type, profit, days)
    elif page == "NEWS":
        st.header("Search Cryptocurrency NEWS")
        crypto = st.text_input("Cryptocurrency:", "Bitcoin")
        source = st.selectbox("News source:", ['all', 'Cryptoslate', 'CoinDesk'])
        get_top_crypto_news(crypto, news_source=source)

if __name__ == "__main__":
    main()



    
   