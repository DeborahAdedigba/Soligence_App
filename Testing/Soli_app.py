# importing necessary modules
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from ta.trend import SMAIndicator
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import os
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
from training import train_all_models
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# Initialize session state
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False
if 'model_paths' not in st.session_state:
    st.session_state.model_paths = {}

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
                use_column_width=True)
        
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

def evaluate_models_selected_coin(coin_index, chosen_model='all'):
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
    
    models = {
        'GRADIENT BOOSTING': GradientBoostingRegressor(),
        'SVR': SVR(),
        'XGBOOST': XGBRegressor(),
        'LSTM': Sequential([LSTM(units=50, input_shape=(X_train.shape[1], 1)), Dense(units=1)])
    }
    
    if chosen_model.lower() == 'all':
        chosen_models = models.keys()
    else:
        chosen_models = [chosen_model.upper()]
    
    eval_metrics = {}
    for model_name in chosen_models:
        if model_name not in models:
            st.warning(f"Model '{model_name}' not found.")
            continue
            
        model = models[model_name]
        model_filename = f"Model_SELECTED_COIN_{coin_index+1}/{model_name.lower().replace(' ', '_')}_model.pkl"
        
        if os.path.exists(model_filename):
            if model_name == 'LSTM':
                model = load_model(model_filename.replace('.pkl', '.h5'))
                X_test_array = X_test.to_numpy().reshape(X_test.shape[0], X_test.shape[1], 1)
                predictions = model.predict(X_test_array).flatten()
            else:
                model = joblib.load(model_filename)
                predictions = model.predict(X_test)
            
            mae = mean_absolute_error(y_test, predictions)
            mse = mean_squared_error(y_test, predictions)
            rmse = np.sqrt(mse)
            mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100
            r2 = r2_score(y_test, predictions)
            
            eval_metrics[model_name] = {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'MAPE': mape, 'R2': r2}
    
    st.subheader(f"Evaluation Metrics for {coin_name}:")
    for model_name, metrics in eval_metrics.items():
        st.write(f"{model_name}:")
        st.write(pd.DataFrame.from_dict(metrics, orient='index', columns=['Value']))
        st.write('---')
    
    if eval_metrics:
        metrics_df = pd.DataFrame.from_dict(eval_metrics, orient='index')
        st.bar_chart(metrics_df[['MAE', 'MSE', 'RMSE']])

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
    
    model_filename = f"Model_SELECTED_COIN_{coin_index+1}/"
    if model_choice == 'LSTM':
        model_filename += "lstm_model.h5"
        if os.path.exists(model_filename):
            model = load_model(model_filename)
            X_array = X.to_numpy().reshape(X.shape[0], X.shape[1], 1)
            predictions = model.predict(X_array[-num_periods:]).flatten()
        else:
            st.error("LSTM model not found.")
            return
    else:
        model_filename += f"{model_choice.lower()}_model.pkl"
        if os.path.exists(model_filename):
            model = joblib.load(model_filename)
            predictions = model.predict(X[-num_periods:])
        else:
            st.error(f"{model_choice} model not found.")
            return
    
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

def apply_ma_trading_strategy(chosen_coin):
    selected_data = combined_data[combined_data['Crypto'] == chosen_coin].copy()
    selected_data.dropna(subset=['Close'], inplace=True)
    selected_data.index = selected_data.index.tz_localize(None)
    
    ma_7 = 7
    ma_14 = 14
    selected_data[f'MA_{ma_7}'] = SMAIndicator(close=selected_data['Close'], window=ma_7).sma_indicator()
    selected_data[f'MA_{ma_14}'] = SMAIndicator(close=selected_data['Close'], window=ma_14).sma_indicator()
    
    selected_data['Buy_Signal'] = np.where(selected_data[f'MA_{ma_7}'] > selected_data[f'MA_{ma_14}'].shift(1), 1, 0)
    selected_data['Sell_Signal'] = np.where(selected_data[f'MA_{ma_7}'] < selected_data[f'MA_{ma_14}'].shift(1), -1, 0)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=selected_data.index, y=selected_data['Close'], name='Close Price', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=selected_data.index, y=selected_data[f'MA_{ma_7}'], name=f'{ma_7}-day MA', line=dict(color='green')))
    fig.add_trace(go.Scatter(x=selected_data.index, y=selected_data[f'MA_{ma_14}'], name=f'{ma_14}-day MA', line=dict(color='red')))
    fig.add_trace(go.Scatter(x=selected_data[selected_data['Buy_Signal'] == 1].index, 
                           y=selected_data[selected_data['Buy_Signal'] == 1]['Close'],
                           mode='markers', name='Buy Signal',
                           marker=dict(color='green', size=10, symbol='triangle-up')))
    fig.add_trace(go.Scatter(x=selected_data[selected_data['Sell_Signal'] == -1].index, 
                           y=selected_data[selected_data['Sell_Signal'] == -1]['Close'],
                           mode='markers', name='Sell Signal',
                           marker=dict(color='red', size=10, symbol='triangle-down')))
    
    current_price = selected_data['Close'].iloc[-1]
    fig.add_trace(go.Scatter(x=[selected_data.index[0], selected_data.index[-1]], 
                           y=[current_price, current_price],
                           mode='lines', name='Current Price',
                           line=dict(color='gray', dash='dash')))
    
    fig.update_layout(title=f'Trading Strategy for {chosen_coin}',
                    xaxis_title='Date',
                    yaxis_title='Price')
    st.plotly_chart(fig)
    
    st.write(f"Current Price: {current_price}")
    return selected_data

def forecast_price(selected_data, chosen_coin, num_days):
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

def determine_best_time_to_trade_future(chosen_coin, num_days):
    selected_data = apply_ma_trading_strategy(chosen_coin)
    future_price, future_date = forecast_price(selected_data, chosen_coin, num_days)
    
    if future_price is not None:
        current_price = selected_data['Close'].iloc[-1]
        action = "Buy" if future_price > current_price else "Sell"
        st.write(f"Recommendation: {action}")
        st.write(f"Forecasted price for {future_date.date()}: {future_price}")
    else:
        st.write("Unable to forecast price.")
    
    return selected_data

def forecast_price_with_model(chosen_coin, num_days, model_type):
    coin_index = selected_data.columns.get_loc(chosen_coin) + 1
    model_filename = f"Model_SELECTED_COIN_{coin_index}/"
    
    if model_type == "SVR":
        model_filename += "svr_model.pkl"
    elif model_type == "GBR":
        model_filename += "gradient_boosting_model.pkl"
    elif model_type == "XGBoost":
        model_filename += "xgboost_model.pkl"
    elif model_type == "LSTM":
        model_filename += "lstm_model.h5"
    else:
        st.error("Invalid model type.")
        return None, None
    
    if not os.path.exists(model_filename):
        st.error(f"Model not found: {model_filename}")
        return None, None
    
    if model_type == "LSTM":
        model = load_model(model_filename)
        features = [f'{chosen_coin}_lag_{lag}' for lag in range(1, 4)]
        X_array = selected_data[features].to_numpy()
        X_today = X_array[-1].reshape(1, 3, 1)
        future_price = model.predict(X_today)[0][0]
    else:
        model = joblib.load(model_filename)
        features = [f'{chosen_coin}_lag_{lag}' for lag in range(1, 4)]
        X_array = selected_data[features].to_numpy()
        X_today = X_array[-1].reshape(1, -1)
        future_price = model.predict(X_today)[0]
    
    future_date = datetime.now() + timedelta(days=num_days)
    return future_price, future_date

def determine_best_time_to_trade(chosen_coin, num_days, model_type):
    selected_data = apply_ma_trading_strategy(chosen_coin)
    future_price, future_date = forecast_price_with_model(chosen_coin, num_days, model_type)
    
    if future_price is not None:
        current_price = selected_data['Close'].iloc[-1]
        action = "Buy" if future_price > current_price else "Sell"
        st.write(f"Recommendation ({model_type}): {action}")
        st.write(f"Forecasted price for {future_date.date()}: {future_price}")
    else:
        st.write("Unable to forecast price.")
    
    return selected_data

def find_best_coins(model_type, desired_profit, num_days):
    if selected_data.empty:
        st.error("No selected coins data available.")
        return
    
    coins = selected_data.columns[:4]
    models = {}
    
    for coin_index, coin in enumerate(coins, start=1):
        model_folder = f"Model_SELECTED_COIN_{coin_index}"
        model_file = f"{model_folder}/{model_type.lower()}_model.pkl"
        
        if os.path.exists(model_file):
            if model_type == 'LSTM':
                models[coin] = tf.keras.models.load_model(model_file.replace('.pkl', '.h5'))
            else:
                models[coin] = joblib.load(model_file)
    
    closest_coin = None
    closest_profit = None
    next_best_coin = None
    next_best_profit = None
    
    for coin, model in models.items():
        input_data = np.array([[num_days, 0, 0]])
        
        if model_type == 'LSTM':
            input_data = input_data.reshape(1, input_data.shape[1], 1)
            price_change = model.predict(input_data)[0][0]
        else:
            price_change = model.predict(input_data)[0]
        
        potential_profit = price_change * desired_profit
        
        if closest_coin is None or abs(potential_profit - desired_profit) < abs(closest_profit - desired_profit):
            next_best_coin = closest_coin
            next_best_profit = closest_profit
            closest_coin = coin
            closest_profit = potential_profit
        elif next_best_coin is None or abs(potential_profit - desired_profit) < abs(next_best_profit - desired_profit):
            next_best_coin = coin
            next_best_profit = potential_profit
    
    st.subheader("Results:")
    if closest_coin:
        st.write(f"Closest coin: {closest_coin} with profit: {closest_profit}")
    if next_best_coin:
        st.write(f"Next best coin: {next_best_coin} with profit: {next_best_profit}")

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
                    use_column_width=True)
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
                                           ["Dataset", "Training Model Metrics", "Prediction Graphs",
                                            "Buy and Sell Prediction", "Predict coin by Profit"])
        
        if prediction_option == "Dataset":
            display_selected_coins()
            plot_coin_scatter()
        elif prediction_option == "Training Model Metrics":
            coins = st.multiselect("Select coins:", selected_data.columns)
            model = st.selectbox("Select model:", ['all', 'Gradient Boosting', 'SVR', 'XGBoost', 'LSTM'])
            
            for coin in coins:
                coin_index = selected_data.columns.get_loc(coin)
                evaluate_models_selected_coin(coin_index, model)
        elif prediction_option == "Prediction Graphs":
            coin = st.selectbox("Select coin:", selected_data.columns)
            model = st.selectbox("Select model:", ['GBR', 'SVR', 'XGB', 'LSTM'])
            frequency = st.selectbox("Select frequency:", ['daily', 'weekly', 'monthly', 'quarterly'])
            periods = st.number_input("Number of periods:", min_value=1, value=20)
            
            if st.button("Predict"):
                coin_index = selected_data.columns.get_loc(coin)
                evaluate_and_plot_model(coin_index, model, frequency, periods)
        elif prediction_option == "Buy and Sell Prediction":
            strategy = st.sidebar.radio("Strategy:", ["Moving Averages", "Models"])
            coin = st.selectbox("Select coin:", selected_data.columns)
            days = st.number_input("Days ahead:", min_value=1, value=10)
            
            if strategy == "Moving Averages":
                determine_best_time_to_trade_future(coin, days)
            else:
                model = st.selectbox("Select model:", ["SVR", "GBR", "XGBoost", "LSTM"])
                determine_best_time_to_trade(coin, days, model)
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