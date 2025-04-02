# importing necceassary modules 
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from ta.trend import SMAIndicator
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA  # Adding PCA import
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
import tensorflow as tf
from training import train_all_models
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf  
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.models import load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import streamlit as st
import plotly.graph_objects as go
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from ta.trend import SMAIndicator
from datetime import datetime, timedelta




# building functions for use in the app
# home page

def home_section():
    st.title("SOLiGence")
    st.write(
        "Solent Intelligence (SOLiGence) is a leading financial multinational organisation that deals"
        " with stock and shares, saving and investments.")

    # Streamlit app
    st.header('Welcome to SOLiGence')
    st.write("Your Intelligent Coin Trading Platform")

    # Brief Description
    st.write("Empower your cryptocurrency trading decisions with AI-driven insights and real-time data.")

    # Call to Action (CTA) Button
    if st.button("Get Started"):
        st.write("Let's explore the world of cryptocurrency trading together!")

        # Background Image
        st.image('sal.jpg', use_container_width=True)
        if 'section' not in st.session_state:
            st.session_state.section = "Home"

        # News and Updates
        st.write("News and Updates:")
        st.info("Stay tuned for the latest updates and trends in the cryptocurrency market!")

        # Testimonials
        st.write("What Our Users Say:")
        st.write("The SOLiGence app transformed how I approach cryptocurrency trading. Highly recommended!")

        # Contact Information
        st.write("Contact Us:")
        st.write("For inquiries, email us at info@soligence.com")

        # Social Media Links
        st.write("Connect with Us:")
        st.markdown("[Twitter](https://twitter.com/SOLiGenceApp) [LinkedIn](https://linkedin.com/company/soligence)")

        # Privacy and Disclaimer
        st.write("Privacy Policy | Disclaimer")


# about the app

def about_us():
    st.title("About Solent Intelligence Ltd.")
    st.write(
        "The scale of this organization's operation is impressive, with millions of subscribers and over 150 billion "
        "pounds worth of investments. This emphasizes the substantial influence that data-driven decisions can have "
        "on managing such a significant amount of assets. The app's focus on implementing an Intelligent Coin Trading "
        "(IST) platform, specifically tailored for crypto coin predictions, resonates deeply with me. The idea of "
        "utilizing AI to predict cryptocurrency prices on different timeframes, such as daily, weekly, monthly, "
        "and quarterly, truly piques my interest. This approach aligns perfectly with my desire to explore how AI can "
        "shape and enhance our daily lives. \n ")
    st.write("The app's ability to recommend trading opportunities by analyzing "
        "AI-generated predictions showcases the tangible applications of data science in the financial world. "
        "Considering a more neutral perspective, while the concept of the app is exciting, there are potential "
        "challenges that need to be acknowledged. Cryptocurrency markets are known for their volatility, and even the "
        "most sophisticated AI predictions might not always be entirely accurate. Users relying solely on these "
        "predictions could face risks if market conditions change unexpectedly.")

    
# getting the real-time data
# Function to fetch cryptocurrency data
def get_crypto_data(ticker, start_date, end_date):
    try:
        crypto = yf.Ticker(ticker)
        data = crypto.history(start=start_date, end=end_date)
        return data
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
        return None

# Example ticker symbols for cryptocurrencies
ticker_symbols = ['BTC-GBP', 'ETH-GBP', 'USDT-GBP', 'BNB-GBP', 'SOL-GBP', 'XRP-GBP', 'USDC-GBP', 'ADA-GBP', 'DOGE-GBP', 'XMR-GBP', 'TRX-GBP', 'DOT-GBP', 'LINK-GBP', 'MATIC-GBP', 'DAI-GBP', 'HBAR-GBP', 'ICP-GBP', 'LTC-GBP', 'BCH-GBP', 'ATOM-GBP', 'ETC-GBP', 'XLM-GBP', 'MKR-GBP', 'TUSD-GBP', 'HEX-GBP', 'XCH-GBP', 'FTM-GBP', 'AXS-GBP', 'NEO-GBP', 'SAND-GBP']

# Define start and end dates for data retrieval (four years ago from today)
end_date = datetime.now()
start_date = end_date - timedelta(days=4*365)  # four years ago

# Fetch data for each cryptocurrency and combine into a single DataFrame
combined_data = pd.DataFrame()
for ticker in ticker_symbols:
    data = get_crypto_data(ticker, start_date, end_date)
    if data is not None:
        data['Crypto'] = ticker  # Add ticker column
        combined_data = pd.concat([combined_data, data], axis=0)

# Drop the 'Dividends' and 'Stock Splits' columns
combined_data.drop(['Dividends', 'Stock Splits'], axis=1, inplace=True)

# Save the combined data to a CSV file
combined_data.to_csv("Cleaned_combined_crypto_data.csv")

# dataset page
def dataset():
    get_crypto_data(ticker, start_date, end_date)
    
    st.header("Crypto Dataset of 30 Coins")
    
    # Sidebar options
    st.sidebar.subheader("Dataset Options")
    
    # Sorting
    sort_column = st.sidebar.multiselect("Sort by:", data.columns)
    ascending = st.sidebar.checkbox("Ascending")
    sorted_data = combined_data.sort_values(by=sort_column, ascending=ascending)

    # Filtering
    selected_crypto = st.sidebar.selectbox("Filter by cryptocurrency:", ['All'] + list(combined_data['Crypto'].unique()))
    if selected_crypto == 'All':
        sorted_data = combined_data
    else:
        sorted_data = sorted_data[sorted_data['Crypto'] == selected_crypto]

    # Pagination
    page_size = st.sidebar.number_input("Items per page:", min_value=1, value=10)
    page_number = st.sidebar.number_input("Page number:", min_value=1, value=1)
    start_idx = (page_number - 1) * page_size
    end_idx = start_idx + page_size
    paginated_data = sorted_data.iloc[start_idx:end_idx]

    # Display the dataset using a dataframe
    st.subheader("Dataset Overview")
    st.dataframe(paginated_data)
    
    # Provide an option to show a table view
    show_table = st.checkbox("Show as Table")
    
    # Display the dataset as a table if the checkbox is selected
    if show_table:
        st.subheader("Dataset Table View")
        st.table(paginated_data)


# plotting the average price trend    
# Function to plot the average price trend
def plot_average_price_trend(data, selected_coin, interval):
   
    # Filter the data for the selected cryptocurrency
    selected_coin_data = data[data['Crypto'] == selected_coin]

    # Check if data for the selected cryptocurrency is present
    if selected_coin_data.empty:
        st.error("No data available for the selected cryptocurrency.")
        return

    # Set 'Date' column as index
    selected_coin_data.index = pd.to_datetime(selected_coin_data.index)

    # Resample the data based on the specified interval and calculate the mean
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
        st.error("Invalid interval. Please choose either 'Daily', 'Weekly', or 'Monthly'.")
        return

    # Plot the trend using date index and average price
    plt.figure(figsize=(10, 6))
    plt.plot(resampled_data.index.date, resampled_data, color='blue', marker='o', linestyle='-')  

    # Adding labels and title
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Average Price', fontsize=14)
    plt.title(f'Average {interval_label} Price Trend for {selected_coin}', fontsize=16)

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')

    # Grid lines
    plt.grid(True, linestyle='--', alpha=0.7)

    # Show plot
    plt.tight_layout()
    st.pyplot(plt)



# volume volatility
def plot_crypto_volatility():
    st.title("Cryptocurrency Volume Volatility")
    
    # Load the cryptocurrency data
    crypto_data = pd.read_csv("Cleaned_combined_crypto_data.csv")
    crypto_data['Date'] = pd.to_datetime(crypto_data['Date'])  
    crypto_data.set_index('Date', inplace=True)  
    
    # Create a selection box for the cryptocurrencies
    available_coins = crypto_data['Crypto'].unique()
    selected_coins = st.multiselect("Select the cryptocurrencies you want to analyze:", options=available_coins, default=available_coins[:2])

    if selected_coins:
        # Filter data for selected cryptocurrencies
        filtered_data = crypto_data[crypto_data['Crypto'].isin(selected_coins)]
        
        # Plotting volume volatility (moving standard deviation of volume) for each selected coin
        window_size = 7  #  window size for rolling calculation
        fig = go.Figure(layout_title_text="Volume Volatility Over Time", layout_xaxis_title="Date", layout_yaxis_title="Volume Volatility (Standard Deviation)")

        for coin in selected_coins:
            # Calculate rolling standard deviation of volume
            coin_data = filtered_data[filtered_data['Crypto'] == coin]['Volume'].rolling(window=window_size).std()
            fig.add_trace(go.Scatter(x=coin_data.index, y=coin_data, mode='lines', name=coin))

        st.plotly_chart(fig)
        

# Function to plot distribution and trend line for selected coin
def plot_distribution_and_trend(selected_coin):
    # Filter data for selected coin
    selected_data = combined_data[combined_data['Crypto'] == selected_coin]
    
    # Create histogram with Plotly
    fig = go.Figure()

    # Add histogram trace
    fig.add_trace(go.Histogram(x=selected_data['Close'], nbinsx=30,
                               histnorm='probability density', marker_color='skyblue',
                               opacity=0.7, name='Histogram'))

    # Calculate KDE for trend line
    kde = gaussian_kde(selected_data['Close'], bw_method=0.2)
    kde_x = np.linspace(selected_data['Close'].min(), selected_data['Close'].max(), 100)
    kde_y = kde.evaluate(kde_x)

    # Add KDE as trend line
    fig.add_trace(go.Scatter(x=kde_x, y=kde_y, mode='lines', 
                             name='Trend Line', line=dict(color='orange')))

    # Customize layout
    fig.update_layout(title=f'Distribution of Close Prices for {selected_coin}',
                      xaxis_title='Close Price', yaxis_title='Probability Density')

    # Show the plot
    st.plotly_chart(fig)

# Plot daily price change
# Function to plot daily price changes for selected coin
def plot_daily_price_changes(selected_coin):
    # Filter data for selected coin
    selected_data = combined_data[combined_data['Crypto'] == selected_coin]
    
    # Calculate daily price changes
    selected_data['Price Change'] = selected_data['Close'].diff()
    
    # Plot daily price changes
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=selected_data.index, y=selected_data['Price Change'], 
                             mode='lines', name='Price Change', line=dict(color='blue')))
    
    # Customize layout
    fig.update_layout(title=f'Daily Price Changes for {selected_coin}',
                      xaxis_title='Date', yaxis_title='Price Change',
                      width=800, height=500)
    
    # Show the plot
    st.plotly_chart(fig)


# boxplot for coins
def plot_boxplot(data, selected_coin):
   
    data= pd.read_csv("Cleaned_combined_crypto_data.csv", index_col = "Date")
    # Filter data for the selected coin
    coin_data = data[data['Crypto'] == selected_coin]

    # Define boxplot data
    boxplot_data = [coin_data['Low'], coin_data['Close'], coin_data['Open'], coin_data['High']]

    # Create boxplot figure using Plotly
    fig = go.Figure()

    # Add box plots for each metric
    for i, metric in enumerate(['Low', 'Close', 'Open', 'High']):
        fig.add_trace(go.Box(y=coin_data[metric], name=metric))

    # Update layout
    fig.update_layout(title=f"Boxplot of High, Low, Open, Close for {selected_coin}",
                      xaxis_title="Metrics",
                      yaxis_title="Price",
                      xaxis=dict(tickvals=[0, 1, 2, 3], ticktext=['Low', 'Close', 'Open', 'High']),
                      showlegend=True)

    # Show plot 
    st.plotly_chart(fig)

# price volume movement
def visualize_crypto_data(combined_data):

    # Get the list of unique coins from the 'Crypto' column
    unique_coins = combined_data['Crypto'].unique()

    # user input of coin name
    selected_coin = st.selectbox("Select the cryptocurrencies you want to analyze:", unique_coins)

    # Check if the entered coin name is valid
    if selected_coin not in unique_coins:
        st.write("Invalid coin name. Please select a valid coin.")
        st.stop()

    # Filter the data for the selected coin
    selected_coin_data = combined_data[combined_data['Crypto'] == selected_coin]

    # Create figure
    fig = go.Figure()

    # Add trace for Close prices on the right y-axis
    fig.add_trace(go.Scatter(x=selected_coin_data.index, y=selected_coin_data['Close'], mode='lines', name='Close', line=dict(color='orange'), yaxis='y2'))

    # Add trace for trading volume on the left y-axis
    fig.add_trace(go.Scatter(x=selected_coin_data.index, y=selected_coin_data['Volume'], fill='tozeroy', name='Volume', line=dict(color='gray', width=0), fillcolor='rgba(128, 128, 128, 0.3)'))

    # Update layout
    fig.update_layout(
        title=f'Close Price and Volume Movement Over Time for {selected_coin}',
        xaxis=dict(title='Date'),
        yaxis=dict(title='Volume', side='left', showgrid=False),
        yaxis2=dict(title='Price', side='right', overlaying='y', showgrid=False),
        legend=dict(x=0, y=1, traceorder="normal"),
    )

    # Show the plot
    st.plotly_chart(fig)



 
    
# pivoting data for easy use of the function
def pivot_data(data):
   
    pivoted_data = data.pivot( columns='Crypto', values='Close')
    return pivoted_data

# calculating and displaying correlated coins
def analyze_coin_correlation(data):
   
    st.header("Coin Correlation Analysis")
    st.write("Streamlit interface for analyzing and displaying the top four positively and negatively correlated cryptocurrencies with user-selected coins.")

    coins_list = data.columns.tolist()
    
    coin_selected = st.selectbox("Select the coin you want to analyze:", coins_list)

    selected_coin_prices = data[coin_selected]

    # Calculate correlations with the selected coin
    correlations = data.corrwith(selected_coin_prices)

    # Sort correlations in descending order
    sorted_correlations = correlations.sort_values(ascending=False)

    # Exclude the selected coin itself for positive and negative correlations
    sorted_correlations = sorted_correlations.drop(coin_selected)

    positively_correlated = sorted_correlations.head(4)
    negatively_correlated = sorted_correlations.tail(4)

    st.write(f"Top four positively correlated cryptocurrencies with {coin_selected}:")
    st.write(positively_correlated)

    st.write(f"\nTop four negatively correlated cryptocurrencies with {coin_selected}:")
    st.write(negatively_correlated)


# ploting the moving average      
def plot_moving_average(crypto_data):
   
    st.header("Moving Average Analysis")
    st.write("Plots the moving average for a user-selected cryptocurrency.")

    # Get the list of available cryptocurrencies
    available_coins = crypto_data['Crypto'].unique()

    # user input for selecting the coin
    coin_selected = st.selectbox("Select the cryptocurrency you want to analyze:", available_coins)

    # User input for selecting the window size
    window_size = st.radio("Select the window size for the moving average:", ['Short (30-day MA)', 'Medium (60-day MA)', 'Long (90-day MA)'])

    # Map window size selection to the corresponding window value
    window_mapping = {'Short (30-day MA)': 30, 'Medium (60-day MA)': 60, 'Long (90-day MA)': 90}
    window = window_mapping[window_size]

    # Filter data for the selected coin
    selected_data = crypto_data[crypto_data['Crypto'] == coin_selected].copy()

    # Calculate the moving average based on the selected window size
    selected_data['MA'] = selected_data['Close'].rolling(window=window).mean()
    window_title = f'{window}-day MA'

    # Plotting
    st.subheader(f"Price Chart for {coin_selected} with {window_title}")
    plt.figure(figsize=(10, 6))
    plt.plot(selected_data.index, selected_data['Close'], label=f'{coin_selected} Price')
    plt.plot(selected_data.index, selected_data['MA'], label=window_title)
    plt.title(f'{coin_selected} Price and Moving Average')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    st.pyplot(plt)

        



# compare coin by first metrics
def plot_crypto_metrics(data, coin, metrics, start_date, end_date):
    
    # check 'Date' column is in datetime format and set it as the index
    if 'Date' in data.columns:
        data['Date'] = pd.to_datetime(data['Date'])
        data.set_index('Date', inplace=True)
    
    plt.figure(figsize=(10, 6))
    
    # Convert start_date and end_date to datetime objects and set timezone to UTC
    start_date = pd.to_datetime(start_date).tz_localize('UTC')
    end_date = pd.to_datetime(end_date).tz_localize('UTC')
    
    # Filter data for the specified coin and date range
    coin_data = data[(data['Crypto'] == coin) & (data.index >= start_date) & (data.index <= end_date)]
    
    
    if coin_data.empty:
        print(f"No data available for {coin} within the specified date range.")
        return
    
    for metric in metrics:
        plt.plot(coin_data.index, coin_data[metric], label=f"{coin} {metric}")
    plt.title(f'Cryptocurrency Prices for {coin} ({start_date} to {end_date})')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    st.pyplot(plt)
    
    
# comparing close prices of Coins

def plot_crypto_coins(data, coins, metric, start_date, end_date):
    # Ensure 'Date' column is in datetime format and set it as the index
    if 'Date' in data.columns:
        data['Date'] = pd.to_datetime(data['Date'])
        data.set_index('Date', inplace=True)
        
    plt.figure(figsize=(10, 6))
    # Convert start_date and end_date to datetime objects and set timezone to UTC
    start_date = pd.to_datetime(start_date).tz_localize('UTC')
    end_date = pd.to_datetime(end_date).tz_localize('UTC')
    
    # Filter data for the specified coin and date range
    for coin in coins:
        coin_data = data[(data['Crypto'] == coin) & (data.index >= start_date) & (data.index <= end_date)]
        plt.plot(coin_data.index, coin_data[metric], label=f"{coin} {metric}")
    plt.title(f'Cryptocurrency {metric} Prices for {", ".join(coins)} ({start_date} to {end_date})')
    plt.xlabel('Date')
    plt.ylabel(metric)
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    st.pyplot(plt)



# Candlestick and volumne of the data

def plot_candlestick_chart(coin='BTC-GBP', period='D'):
    # Filter the DataFrame for the selected coin
    coin_data = crypto_data[crypto_data['Crypto'] == coin].copy()
    
    # Make sure the index is a DatetimeIndex
    coin_data.index = pd.to_datetime(coin_data.index)
    
    # Resample data based on the selected period
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
    else:  # Default to daily data if period is not weekly or monthly
        resampled_data = coin_data

    # Plotting
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
                      yaxis_title='Price',
                      xaxis_rangeslider_visible=False)

    st.plotly_chart(fig)

# visualizing market state

def visualize_market_state(data):

    # User input for selecting cryptocurrencies
    st.sidebar.title("Market State Prediction")
    available_coins = data['Crypto'].unique()
    coin_selected = st.sidebar.selectbox("Select cryptocurrencies to analyze", available_coins)

    # Filter data for the selected coins
    selected_data = data[data['Crypto'] == coin_selected].copy()
    
    # Convert 'Close' column to numeric, dropping rows with non-numeric values
    selected_data['Close'] = pd.to_numeric(selected_data['Close'], errors='coerce')
    selected_data.dropna(subset=['Close'], inplace=True)

    # Calculate daily price changes for each coin
    price_changes = selected_data['Close'].pct_change()

    # Replace infinite values with NaN and drop rows with NaN values
    price_changes.replace([np.inf, -np.inf], np.nan, inplace=True)
    price_changes.dropna(inplace=True)

    # Predict market state (up or down) based on overall price movement of selected coins
    market_state = np.sign(price_changes).astype(float)
    
    # Display explanation
    st.write("\nExplanation:")
    st.write("Down: A value of -1.0 suggests that the market is predicted to decrease or go down on that particular day.")
    st.write("UP: A value of 1.0 suggests that the market is predicted to increase or go up on that particular day.")

    # Visualize market state
    fig, ax = plt.subplots(figsize=(10, 6))
    selected_data['Close'].plot(ax=ax, label='Close Price')
    ax.scatter(selected_data.index, selected_data['Close'], c='g', label='Up Market', marker='^', alpha=0.5)
    ax.scatter(selected_data.index, selected_data['Close'], c='r', label='Down Market', marker='v', alpha=0.5)
    ax.set_title(f'{coin_selected} Market State Prediction')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend()
    plt.show()
    st.pyplot(fig)
    
# visualising the high and low
def predict_highs_lows(data):
   
    # Extract available coins from the data
    available_coins = data['Crypto'].unique()

    # User input for selecting the coin
    coin_selected = st.sidebar.selectbox("Select the cryptocurrency to analyze", available_coins)

    # Filter data for the selected coin
    selected_data = data[data['Crypto'] == coin_selected].copy()

    # Calculate percentiles for highs and lows
    high_threshold = np.percentile(selected_data['High'], 90)
    low_threshold = np.percentile(selected_data['Low'], 10)

    # Predict highs and lows based on thresholds
    predicted_highs = selected_data['Close'] > high_threshold
    predicted_lows = selected_data['Close'] < low_threshold

    # Visualize predicted highs and lows
    st.title(f'Predicted Highs and Lows for {coin_selected}')
    plt.figure(figsize=(10, 6))
    plt.plot(selected_data.index, selected_data['Close'], label='Close Price')
    plt.scatter(selected_data.index[predicted_highs], selected_data['Close'][predicted_highs], color='red', label='Predicted Highs')
    plt.scatter(selected_data.index[predicted_lows], selected_data['Close'][predicted_lows], color='green', label='Predicted Lows')
    plt.title(f'Predicted Highs and Lows for {coin_selected}')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    st.pyplot(plt)

    # Format output as DataFrame
    predicted_highs_df = pd.DataFrame({'Date': selected_data.index, 'Predicted Highs': predicted_highs})
    predicted_lows_df = pd.DataFrame({'Date': selected_data.index, 'Predicted Lows': predicted_lows})

    return predicted_highs_df, predicted_lows_df
    


# generating the four coins from through pca and clustering
def generate_selected_data(combined_data):
    
    # Pivot and preprocess the data
    pivoted_data = combined_data.pivot(columns='Crypto', values='Close')
    pivoted_data_filled = pivoted_data.fillna(0)  # Create a new DataFrame without modifying the original

    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(pivoted_data_filled)


    # Perform PCA with a reasonable number of components
    pca = PCA(n_components=10)  # Adjusted for example, consider your variance ratio to decide
    pca_result = pca.fit_transform(scaled_data)

    # Compute component loadings (transposed PCA components)
    loadings = pd.DataFrame(pca.components_.T, columns=[f'PC{i}' for i in range(1, 11)], index=pivoted_data.columns)

    # Perform K-means clustering on the loadings
    kmeans = KMeans(n_clusters=4, random_state=0)
    cluster_labels = kmeans.fit_predict(loadings)

    # Assign cluster labels to the coins
    loadings['Cluster'] = cluster_labels

    # Select a representative coin from each cluster
    representative_coins = pd.DataFrame()
    for i in range(4):
        cluster = loadings[loadings['Cluster'] == i]
        # Calculate the distance to the cluster center
        center = kmeans.cluster_centers_[i]
        cluster['distance_to_center'] = cluster.apply(lambda x: np.linalg.norm(x[:-1] - center), axis=1)
        # Append the coin with the minimum distance to the center
        representative_coins = pd.concat([representative_coins, cluster.loc[[cluster['distance_to_center'].idxmin()]]])


    # Extract the selected coins based on the index of representative_coins
    selected_data = pivoted_data[representative_coins.index]

    return selected_data
selected_data = generate_selected_data(combined_data)
# saving the data to csv
selected_data.to_csv('Selected_coins.csv', index=True)


    

# Function to train models in the background when the app starts
def train_models_background(selected_data):
    st.write("Training models in the background...")
    # Your training code here
    st.session_state.models_trained = True
    st.write("Training completed.")

# Check if training has already been done
if 'models_trained' not in st.session_state:
    train_models_background(selected_data)  

    

    
    
    
# boxplot for the four selected coins
def display_selected(selected_data):
    st.title("Boxplot of the 4 selected coins from K-mean clustering")

    # Initialize subplots with 1 row and 4 columns
    fig = make_subplots(rows=1, cols=4)

    # Add box traces for each coin in separate subplots
    for i, (columnName, columnData) in enumerate(selected_data.iteritems(), start=1):
        fig.add_trace(go.Box(y=columnData, name=columnName), row=1, col=i)

    # Update layout
    fig.update_layout(title='Boxplot of the 4 selected coins from K-mean clustering',
                      xaxis_title='Coins',
                      yaxis_title='Price',
                      showlegend=False,
                      width=800, height=400)

    # Show plot
    st.plotly_chart(fig)


# plotting the four coins against each other to see their relationships     
def plot_coin_scatter(selected_data):
    # Get all combinations of coins
    coin_combinations = list(itertools.combinations(selected_data.columns, 2))

    # Create a subplot for each pair of coins
    fig = make_subplots(rows=3, cols=3, subplot_titles=[f"{coin1} vs {coin2}" for coin1, coin2 in coin_combinations])

    # Plot scatter plots for each pair of coins
    for i, (coin1, coin2) in enumerate(coin_combinations, start=1):
        row = (i - 1) // 3 + 1
        col = (i - 1) % 3 + 1
        fig.add_trace(go.Scatter(x=selected_data[coin1], y=selected_data[coin2], mode='markers', name=f"{coin1} vs {coin2}"), row=row, col=col)

    # Update layout
    fig.update_layout(height=600, width=800, showlegend=False)  # Adjust height and width here

    # Show plot
    st.plotly_chart(fig)





# making prediction from the trained and saved models
# Function to evaluate models for selected coins
def evaluate_models_selected_coin(selected_data, column_index, chosen_model='all'):
    coin_name = selected_data.columns[column_index]

    # Add lagged features for 1 to 3 days
    for lag in range(1, 4):
        selected_data.loc[:, f'{coin_name}_lag_{lag}'] = selected_data[coin_name].shift(lag)

    # Drop rows with NaN values created due to shifting
    selected_data.dropna(inplace=True)

    # Features will be the lagged values, and the target will be the current price of the coin
    features = [f'{coin_name}_lag_{lag}' for lag in range(1, 4)]
    X = selected_data[features]
    y = selected_data[coin_name]

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize dictionary to hold models
    models = {
        'GRADIENT BOOSTING': GradientBoostingRegressor(),
        'SVR': SVR(),
        'XGBOOST': XGBRegressor(),
        'LSTM': Sequential([LSTM(units=50, input_shape=(X_train.shape[1], 1)), Dense(units=1)])
    }

    eval_metrics = {}

    if chosen_model.lower() == 'all':
        chosen_models = models.keys()
    else:
        chosen_models = [chosen_model.upper()]  # Capitalize input for case insensitivity

    for model_name in chosen_models:
        if model_name not in models:
            st.warning(f"Model '{model_name}' not found. Skipping...")
            continue

        model = models[model_name]

        if model_name == 'LSTM':
            model_filename = f"Model_SELECTED_COIN_{column_index+1}/lstm_model.h5"
            if os.path.exists(model_filename):
                model = load_model(model_filename)
                # Reshape the input data for LSTM model
                X_test_array = X_test.to_numpy().reshape(X_test.shape[0], X_test.shape[1], 1)
                predictions = model.predict(X_test_array).flatten()
            else:
                st.error("No pre-trained LSTM model found.")
                return  # Skip the rest of the loop if LSTM model not found
        else:
            # Load pre-saved model using pickle
            model_filename = f"Model_SELECTED_COIN_{column_index+1}/{model_name.lower().replace(' ', '_')}_model.pkl"

            if os.path.exists(model_filename):
                with open(model_filename, 'rb') as f:
                    loaded_model = pickle.load(f)  # Use a different variable name
            else:
                st.error(f"No pre-trained {model_name} model found.")
                return  # Skip the rest of the loop if the model not found

            predictions = loaded_model.predict(X_test)

        # Calculate evaluation metrics
        mae = mean_absolute_error(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100
        r2 = r2_score(y_test, predictions)

        # Store evaluation metrics
        eval_metrics[model_name] = {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'MAPE': mape, 'R2': r2}

    # Display evaluation metrics
    st.subheader(f"Evaluation Metrics for {coin_name}:")
    for model_name, metrics in eval_metrics.items():
        st.write(f"Evaluation metrics for {model_name}:")
        for metric_name, value in metrics.items():
            st.write(f"{metric_name}: {value}")
        st.write('---')

    # Plot evaluation metrics
    if eval_metrics:
        st.subheader("Evaluation Metric Visualization")
        fig, ax = plt.subplots(figsize=(12, 6))

        metrics = list(eval_metrics.keys())
        mae_values = [eval_metrics[model]['MAE'] for model in metrics]
        mse_values = [eval_metrics[model]['MSE'] for model in metrics]
        rmse_values = [eval_metrics[model]['RMSE'] for model in metrics]

        bar_width = 0.15
        index = np.arange(len(metrics))

        bar1 = ax.bar(index - 2*bar_width, mae_values, bar_width, label='MAE')
        bar2 = ax.bar(index - bar_width, mse_values, bar_width, label='MSE')
        bar3 = ax.bar(index, rmse_values, bar_width, label='RMSE')

        ax.set_xlabel('Models')
        ax.set_ylabel('Metrics')
        ax.set_title(f'Evaluation Metrics for {coin_name} using {chosen_model.upper()} as the Models')
        ax.set_xticks(index)
        ax.set_xticklabels(metrics)
        ax.legend()

        # Annotate bars with values
        for bars in [bar1, bar2, bar3]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate('{}'.format(round(height, 2)),
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')

        st.pyplot(fig)
    else:
        st.error("No models were evaluated.")


# prediction graphs
def plot_actual_forecast_with_confidence(actual, predictions, periods, upper_bound, lower_bound):
    """
    Plot actual prices and forecasted prices with confidence intervals using Plotly.
    """
    # Create figure
    fig = go.Figure()

    # Add actual prices trace
    fig.add_trace(go.Scatter(x=periods, y=actual, mode='lines', name='Actual Price', line=dict(color='green')))

    # Add forecasted prices trace
    fig.add_trace(go.Scatter(x=periods, y=predictions, mode='lines', name='Forecasted Price', line=dict(color='red')))

    # Add confidence interval
    fig.add_trace(go.Scatter(x=periods, y=upper_bound, mode='lines', name='Upper Bound', fill=None,
                             line=dict(color='blue', width=0)))
    fig.add_trace(go.Scatter(x=periods, y=lower_bound, mode='lines', name='Lower Bound',
                             fill='tonexty', line=dict(color='blue')))

    # Update layout
    fig.update_layout(title="Actual and Forecasted Prices with Confidence Intervals",
                      xaxis_title='Date',
                      yaxis_title='Price',
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                      plot_bgcolor='rgba(0,0,0,0)',
                      showlegend=True,
                      template='plotly_white')

    # Display figure 
    st.plotly_chart(fig)


# Function to evaluate models and plot predictions for each coin
def evaluate_and_plot_model(selected_data, coin_index, model_choice):
    coin_name = selected_data.columns[coin_index]  
    # Add lagged features for 1 to 3 days
    for lag in range(1, 4):
        selected_data[f'{coin_name}_lag_{lag}'] = selected_data[coin_name].shift(lag)

    # Drop rows with NaN values created due to shifting
    selected_data.dropna(inplace=True)

    # Features will be the lagged values, and the target will be the current price of the coin
    features = [f'{coin_name}_lag_{lag}' for lag in range(1, 4)]
    X = selected_data[features]
    y = selected_data[coin_name]

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize dictionary to hold models
    models = {
        'GBR': GradientBoostingRegressor(),
        'SVR': SVR(),
        'XGB': XGBRegressor(), 
        'LSTM': Sequential([LSTM(units=50, input_shape=(X_train.shape[1], 1)), Dense(units=1)])
    }

    # Load pre-trained models for SVR, XGBoost, and Gradient Boosting
    for model_name in ['SVR', 'XGBoost', 'Gradient Boosting']:
        model_filename = f"Model_SELECTED_COIN_{coin_index + 1}/{model_name.lower().replace(' ', '_')}_model.pkl"
        if os.path.exists(model_filename):
            models[model_name] = joblib.load(model_filename)
        else:
            print(f"No pre-trained model found for {model_name}. Skipping...")

    # Initialize and train the selected model
    if model_choice in models:
        model = models[model_choice]
        if model_choice != 'LSTM':
            model.fit(X_train, y_train)
    elif model_choice == 'LSTM':
        model_filename = f"Model_SELECTED_COIN_{coin_index + 1}/lstm_model.h5"
        if os.path.exists(model_filename):
            model = tf.keras.models.load_model(model_filename)
            # Reshape the input data for LSTM model
            X_test_array = X_test.to_numpy().reshape(X_test.shape[0], X_test.shape[1], 1)
            predictions = model.predict(X_test_array).flatten()
        else:
            print("No pre-trained LSTM model found.")
            return None, None, None, None
    else:
        print("Invalid model choice. Please choose from SVR, XGBoost, Gradient Boosting, or LSTM.")
        return None, None, None, None

    # Make predictions for the specified number of periods
    predictions = model.predict(X[-num_periods:])  

    # Get the last date in the dataset
    last_date = selected_data.index[-1]

    # Generate periods for future predictions
    if frequency == 'daily':
        periods = pd.date_range(start=last_date, periods=num_periods, freq='D')
    elif frequency == 'weekly':
        periods = pd.date_range(start=last_date, periods=num_periods, freq='W')
    elif frequency == 'monthly':
        periods = pd.date_range(start=last_date, periods=num_periods, freq='M')
    elif frequency == 'quarterly':
        periods = pd.date_range(start=last_date, periods=num_periods, freq='Q')
    else:
        print("Invalid frequency. Please choose from 'daily', 'weekly', 'monthly', or 'quarterly'.")

    # Calculate mean squared error
    mse = mean_squared_error(y_test[-num_periods:], predictions)

    # Plot actual and forecasted prices with confidence intervals
    upper_bound = predictions + 1.96 * np.sqrt(mse)
    lower_bound = predictions - 1.96 * np.sqrt(mse)

    # Flatten arrays for fill_between
    upper_bound = upper_bound.flatten()
    lower_bound = lower_bound.flatten()

    # Create a DataFrame with dates and predictions
    predictions_df = pd.DataFrame({'Date': periods, 'Predictions': predictions.flatten()})

    # Display the DataFrame in Streamlit
    st.subheader("Predictions with Dates:")
    st.dataframe(predictions_df)

    # Plot the time series plot with averages and confidence intervals using Streamlit's plotting functions
    st.subheader(f"Predicted Prices and Confidence Intervals for {coin_name} by {frequency}")
    plot_actual_forecast_with_confidence(y_test[-num_periods:], predictions, periods, upper_bound, lower_bound)



# predicting Buy and selling
# Function to apply MA Trading Strategy and display chart
def determine_best_time_to_trade_future(selected_data, chosen_coin, num_days):
    # Apply MA trading strategy
    total_profit_loss = apply_ma_trading_strategy(selected_data, chosen_coin)

    # Forecast future price
    future_price, future_date = forecast_price(selected_data, chosen_coin, num_days)

    if future_price is not None:
        action = "Buy" if future_price > selected_data[chosen_coin].iloc[-1] else "Sell"
        st.write(f"Recommended action: {action}")
        st.write(f"Forecasted price for {future_date.date()}: {future_price}")
    else:
        st.write("Unable to forecast price for the future.")

    return total_profit_loss

# Function to apply MA Trading Strategy and display chart
def apply_ma_trading_strategy(selected_data, chosen_coin):
    # Drop rows with missing 'Close' prices
    selected_data.dropna(subset=[chosen_coin], inplace=True)

    # Convert index to offset-naive datetime index
    selected_data.index = selected_data.index.tz_localize(None)

    # Calculate moving averages
    ma_7 = 7  # 7-day moving average
    ma_14 = 14  # 14-day moving average
    selected_data[f'MA_{ma_7}'] = SMAIndicator(close=selected_data[chosen_coin], window=ma_7).sma_indicator()
    selected_data[f'MA_{ma_14}'] = SMAIndicator(close=selected_data[chosen_coin], window=ma_14).sma_indicator()

    # Generate buy and sell signals based on moving averages
    selected_data['Buy_Signal'] = np.where(selected_data[f'MA_{ma_7}'] > selected_data[f'MA_{ma_14}'].shift(1), 1, 0)
    selected_data['Sell_Signal'] = np.where(selected_data[f'MA_{ma_7}'] < selected_data[f'MA_{ma_14}'].shift(1), -1, 0)

    # Create plotly figure
    fig = go.Figure()

    # Add traces
    fig.add_trace(go.Scatter(x=selected_data.index, y=selected_data[chosen_coin], name='Close Price', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=selected_data.index, y=selected_data[f'MA_{ma_7}'], name=f'{ma_7}-day MA', line=dict(color='green')))
    fig.add_trace(go.Scatter(x=selected_data.index, y=selected_data[f'MA_{ma_14}'], name=f'{ma_14}-day MA', line=dict(color='red')))
    fig.add_trace(go.Scatter(x=selected_data[selected_data['Buy_Signal'] == 1].index, y=selected_data[selected_data['Buy_Signal'] == 1][chosen_coin],
                             mode='markers', marker=dict(color='green', size=10, symbol='triangle-up'), name='Buy Signal'))
    fig.add_trace(go.Scatter(x=selected_data[selected_data['Sell_Signal'] == -1].index, y=selected_data[selected_data['Sell_Signal'] == -1][chosen_coin],
                             mode='markers', marker=dict(color='red', size=10, symbol='triangle-down'), name='Sell Signal'))
    
    # Current price line
    current_price = selected_data[chosen_coin].iloc[-1]
    fig.add_trace(go.Scatter(x=[selected_data.index[0], selected_data.index[-1]], y=[current_price, current_price],
                             mode='lines', line=dict(color='gray', dash='dash'), name='Current Price'))

    # Update layout
    fig.update_layout(
        title=f'Moving Average Trading Strategy for {chosen_coin}',
        xaxis_title='Date',
        yaxis_title='Price',
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        width=1000,
        height=600
    )

    # Display plotly figure
    st.plotly_chart(fig)

    st.write(f"Current Price of {chosen_coin} is {current_price}")

    return selected_data

def forecast_price(selected_data, chosen_coin, num_days):
    future_date = datetime.now() + timedelta(days=num_days)
    future_date = future_date.replace(tzinfo=None)  # Remove timezone information

    # Convert index datetime to naive datetime
    index_datetime = selected_data.index[-1].to_pydatetime().replace(tzinfo=None)

    if future_date > index_datetime:
        # Extend the index
        new_index = pd.date_range(start=selected_data.index[0], periods=len(selected_data) + num_days, freq='D')
        extended_data = selected_data.reindex(new_index, method='ffill')

        # Recalculate MAs for the extended period
        ma_7 = 7
        ma_14 = 14
        extended_data[f'MA_{ma_7}'] = SMAIndicator(close=extended_data[chosen_coin], window=ma_7).sma_indicator()
        extended_data[f'MA_{ma_14}'] = SMAIndicator(close=extended_data[chosen_coin], window=ma_14).sma_indicator()

        # Forecast using the latest MA values
        future_price = (extended_data[f'MA_{ma_7}'].iloc[-1] + extended_data[f'MA_{ma_14}'].iloc[-1]) / 2
        return future_price, future_date
    else:
        st.write("Future date is within the available data range.")
        return None, None





# ML on trading strategy

# Function to apply MA Trading Strategy and display chart
def apply_ma_trading_strategy(selected_data, chosen_coin):
    # Drop rows with missing 'Close' prices
    selected_data.dropna(subset=[chosen_coin], inplace=True)

    # Calculate moving averages
    ma_7 = 7
    ma_14 = 14
    selected_data[f'MA_{ma_7}'] = SMAIndicator(close=selected_data[chosen_coin], window=ma_7).sma_indicator()
    selected_data[f'MA_{ma_14}'] = SMAIndicator(close=selected_data[chosen_coin], window=ma_14).sma_indicator()

    # Generate buy and sell signals based on moving averages
    selected_data['Buy_Signal'] = np.where(selected_data[f'MA_{ma_7}'] > selected_data[f'MA_{ma_14}'].shift(1), 1, 0)
    selected_data['Sell_Signal'] = np.where(selected_data[f'MA_{ma_7}'] < selected_data[f'MA_{ma_14}'].shift(1), -1, 0)

    # Display plotly figure
    fig = go.Figure()

    # Add traces
    fig.add_trace(go.Scatter(x=selected_data.index, y=selected_data[chosen_coin], name='Close Price', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=selected_data.index, y=selected_data[f'MA_{ma_7}'], name=f'{ma_7}-day MA', line=dict(color='green')))
    fig.add_trace(go.Scatter(x=selected_data.index, y=selected_data[f'MA_{ma_14}'], name=f'{ma_14}-day MA', line=dict(color='red')))
    fig.add_trace(go.Scatter(x=selected_data[selected_data['Buy_Signal'] == 1].index, y=selected_data[selected_data['Buy_Signal'] == 1][chosen_coin],
                             mode='markers', marker=dict(color='green', size=10, symbol='triangle-up'), name='Buy Signal'))
    fig.add_trace(go.Scatter(x=selected_data[selected_data['Sell_Signal'] == -1].index, y=selected_data[selected_data['Sell_Signal'] == -1][chosen_coin],
                             mode='markers', marker=dict(color='red', size=10, symbol='triangle-down'), name='Sell Signal'))
    
    # Current price line
    current_price = selected_data[chosen_coin].iloc[-1]
    fig.add_trace(go.Scatter(x=[selected_data.index[0], selected_data.index[-1]], y=[current_price, current_price],
                             mode='lines', line=dict(color='gray', dash='dash'), name='Current Price'))

    # Update layout
    fig.update_layout(
        title=f'Moving Average Trading Strategy for {chosen_coin}',
        xaxis_title='Date',
        yaxis_title='Price',
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        width=1000,
        height=600
    )

    # Display plotly figure
    st.plotly_chart(fig)

    st.write(f"Current Price of {chosen_coin} is {current_price}")

    return selected_data

# Function to forecast future price using the SVR model
def forecast_price_svr(selected_data, chosen_coin, num_days):
    coin_index = selected_data.columns.get_loc(chosen_coin) + 1
    model_filename = f"Model_SELECTED_COIN_{coin_index}/svr_model.pkl"
    if not os.path.exists(model_filename):
        st.write(f"No pre-trained SVR model found for {chosen_coin}.")
        return None, None

    # Load the SVR model
    model = joblib.load(model_filename)

    # Prepare input data for prediction
    features = [f'{chosen_coin}_lag_{lag}' for lag in range(1, 4)]
    X_array = selected_data[features].to_numpy()
    X_today = X_array[-1].reshape(1, -1)

    # Predict future price using the SVR model
    future_price = model.predict(X_today)[0]
    future_date = datetime.now() + timedelta(days=num_days)

    return future_price, future_date

# Function to forecast future price using the GBR model
def forecast_price_gbr(selected_data, chosen_coin, num_days):
    coin_index = selected_data.columns.get_loc(chosen_coin) + 1
    model_filename = f"Model_SELECTED_COIN_{coin_index}/gradient_boosting_model.pkl"
    if not os.path.exists(model_filename):
        st.write(f"No pre-trained GBR model found for {chosen_coin}.")
        return None, None

    # Load the GBR model
    model = joblib.load(model_filename)

    # Prepare input data for prediction
    features = [f'{chosen_coin}_lag_{lag}' for lag in range(1, 4)]
    X_array = selected_data[features].to_numpy()
    X_today = X_array[-1].reshape(1, -1)

    # Predict future price using the GBR model
    future_price = model.predict(X_today)[0]
    future_date = datetime.now() + timedelta(days=num_days)

    return future_price, future_date

# Function to forecast future price using the XGBoost model
def forecast_price_xgboost(selected_data, chosen_coin, num_days):
    coin_index = selected_data.columns.get_loc(chosen_coin) + 1
    model_filename = f"Model_SELECTED_COIN_{coin_index}/xgboost_model.pkl"
    if not os.path.exists(model_filename):
        st.write(f"No pre-trained XGBoost model found for {chosen_coin}.")
        return None, None

    # Load the XGBoost model
    model = joblib.load(model_filename)

    # Prepare input data for prediction
    features = [f'{chosen_coin}_lag_{lag}' for lag in range(1, 4)]
    X_array = selected_data[features].to_numpy()
    X_today = X_array[-1].reshape(1, -1)

    # Predict future price using the XGBoost model
    future_price = model.predict(X_today)[0]
    future_date = datetime.now() + timedelta(days=num_days)

    return future_price, future_date

# Function to forecast future price using the LSTM model
def forecast_price_lstm(selected_data, chosen_coin, num_days):
    coin_index = selected_data.columns.get_loc(chosen_coin) + 1
    model_filename = f"Model_SELECTED_COIN_{coin_index}/lstm_model.h5"
    if not os.path.exists(model_filename):
        st.write(f"No pre-trained LSTM model found for {chosen_coin}.")
        return None, None

    # Load the LSTM model
    model = load_model(model_filename)

    # Prepare input data for prediction
    features = [f'{chosen_coin}_lag_{lag}' for lag in range(1, 4)]
    X_array = selected_data[features].to_numpy()
    X_today = X_array[-1].reshape(1, 3, 1)  # LSTM expects input shape (batch_size, timesteps, input_dim)

    # Predict future price using the LSTM model
    future_price = model.predict(X_today)[0][0]
    future_date = datetime.now() + timedelta(days=num_days)

    return future_price, future_date

# Function to determine the best time to trade based on model predictions
def determine_best_time_to_trade(selected_data, chosen_coin, num_days, model):
    if model == "SVR":
        total_profit_loss = apply_ma_trading_strategy(selected_data, chosen_coin)
        future_price, future_date = forecast_price_svr(selected_data, chosen_coin, num_days)
    elif model == "GBR":
        total_profit_loss = apply_ma_trading_strategy(selected_data, chosen_coin)
        future_price, future_date = forecast_price_gbr(selected_data, chosen_coin, num_days)
    elif model == "XGBoost":
        total_profit_loss = apply_ma_trading_strategy(selected_data, chosen_coin)
        future_price, future_date = forecast_price_xgboost(selected_data, chosen_coin, num_days)
    elif model == "LSTM":
        total_profit_loss = apply_ma_trading_strategy(selected_data, chosen_coin)
        future_price, future_date = forecast_price_lstm(selected_data, chosen_coin, num_days)
    else:
        st.write("Invalid model selection.")
        return None

    if future_price is not None:
        action = "Buy" if future_price > selected_data[chosen_coin].iloc[-1] else "Sell"
        st.write(f"Recommended action for {chosen_coin}: {action}")
        st.write(f"Forecasted price for {future_date.date()}: {future_price}")
    else:
        st.write(f"Unable to forecast price for {chosen_coin} in the future.")

    return total_profit_loss





# prediction coin based on profit and num of days

def find_best_coins(selected_data_path):
    # Load selected data
    # selected_data = pd.read_csv(selected_data_path, index_col="Date")

    # Extract coins from column names
    coins = selected_data.columns[:]  # Assuming the first column is not a coin name


    # Load the saved models for the selected model type and each coin
    models = {}
    for coin_index, coin in enumerate(coins, start=1):
        model_folder = os.path.join("Model_SELECTED_COIN_" + str(coin_index))
        model_file = os.path.join(model_folder, f"{model_type.lower()}_model.pkl")
        if os.path.exists(model_file):
            if model_type == 'LSTM':
                models[coin] = tf.keras.models.load_model(model_file)
            else:
                models[coin] = joblib.load(model_file)
        else:
            st.error(f"No pre-trained model found for {coin} and {model_type}.")
            continue


    # Initialize variables to track the closest and next best coins and profits
    closest_coin = None
    closest_profit = None
    next_best_coin = None
    next_best_profit = None

    # Iterate through the coins
    for coin, model in models.items():
        # Reshape the input data to match the shape of the training data
        input_data = np.array([[num_days, 0, 0]])  # Assuming the second and third features are placeholders
        # Predict the price change for the specified number of days
        if model_type == 'LSTM':
            input_data = input_data.reshape(1, input_data.shape[1], 1)  # Reshaping to (batch_size, timesteps, input_dim)
            price_change = model.predict(input_data)[0]
        else:
            price_change = model.predict(input_data)[0]
        # Calculate the potential profit
        potential_profit = price_change * desired_profit
        
        # Update closest and next best coins and profits
        if closest_coin is None or abs(potential_profit - desired_profit) < abs(closest_profit - desired_profit):
            next_best_coin = closest_coin
            next_best_profit = closest_profit
            closest_coin = coin
            closest_profit = potential_profit
        elif next_best_coin is None or abs(potential_profit - desired_profit) < abs(next_best_profit - desired_profit):
            next_best_coin = coin
            next_best_profit = potential_profit
    
    # Display the closest and next best coins and their potential profits
    st.subheader("Results:")
    if closest_coin:
        closest_profit_value = closest_profit[0] if isinstance(closest_profit, np.ndarray) else closest_profit
        st.write(f"The closest coin to yield the desired profit of {desired_profit} GBP in {num_days} days is: {closest_coin}")
        st.write(f"The potential profit for {closest_coin} is: {closest_profit_value}")
    else:
        st.write("No results found for the given parameters.")
    if next_best_coin:
        next_best_profit_value = next_best_profit[0] if isinstance(next_best_profit, np.ndarray) else next_best_profit
        st.write(f"The next best coin is: {next_best_coin}")
        st.write(f"The potential profit for {next_best_coin} is: {next_best_profit_value}")

        
# getting crypto news by scrapping        

def get_top_crypto_news(crypto, num_stories=5, news_source='all'):
    if news_source == 'all' or news_source == 'Cryptoslate':
        # URL of the RSS feed from Cryptoslate
        cryptoslate_feed_url = 'https://cryptoslate.com/feed/'

        # Parse the RSS feed from Cryptoslate
        cryptoslate_feed = feedparser.parse(cryptoslate_feed_url)

        # Filter news stories based on the chosen cryptocurrency from Cryptoslate
        crypto_news_cryptoslate = [entry for entry in cryptoslate_feed.entries if crypto.lower() in entry.title.lower()]

        # Display Cryptoslate news stories
        if crypto_news_cryptoslate:
            st.write(f"Top {num_stories} Cryptoslate news stories about {crypto}:")
            for i, entry in enumerate(crypto_news_cryptoslate[:num_stories], start=1):
                st.write(f"{i}. [{entry.title}]({entry.link}) - {entry.published}")
        else:
            st.write(f"No Cryptoslate news found for {crypto}.")

    if news_source == 'all' or news_source == 'CoinDesk':
        # URL of the RSS feed from CoinDesk
        coindesk_feed_url = 'https://feeds.feedburner.com/CoinDesk'

        # Parse the RSS feed from CoinDesk
        coindesk_feed = feedparser.parse(coindesk_feed_url)

        # Filter news stories based on the chosen cryptocurrency from CoinDesk
        crypto_news_coindesk = [entry for entry in coindesk_feed.entries if crypto.lower() in entry.title.lower()]

        # Display CoinDesk news stories
        if crypto_news_coindesk:
            st.write(f"Top {num_stories} CoinDesk news stories about {crypto}:")
            for i, entry in enumerate(crypto_news_coindesk[:num_stories], start=1):
                st.write(f"{i}. [{entry.title}]({entry.link}) - {entry.published}")
        else:
            st.write(f"No CoinDesk news found for {crypto}.")



        
# App layout

st.markdown("""
<style>
/* Setting the font family and color across all text in the app */
body {
    font-family: 'Arial', sans-serif;
    color: #ffffff;
    background-color: #4B0082;
    font-size: 18px; /* Increased font size */
}

/* Customizing the sidebar background and text color */
.sidebar .sidebar-content {
    background-image: linear-gradient(#6a0dad, #7b68ee);
    color: #ffffff;
}

/* Style adjustments for sidebar labels */
.sidebar .sidebar-content .Widget>label {
    color: #ffffff;
}

/* Background color adjustments for streamlit components */
div.stButton > button:first-child {
    background-color: #6a0dad;
    color: #ffffff;
}

/* Adjusting the report background and font color */
.reportview-container .main {
    background-color: #4B0082;
    color: #ffffff;
}

/* Footer customization */
footer {
    background-color: #6a0dad;
    color: #ffffff;
}

/* Header customization */
header {
    background-color: #4B0082;
}

/* Setting the background image */
[data-testid="stAppViewContainer"] {
    background-image: url("https://img.freepik.com/premium-photo/white-bitcoin-cryptocurrency-coin-against-grey-background-d-rendering_601748-3477.jpg");
    background-size: cover;
}
</style>
""", unsafe_allow_html=True)

import functools

# Create a decorator to modify the behavior of st.write()
def custom_write(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        new_args = []
        for arg in args:
            if isinstance(arg, str):
                # Wrap string arguments in HTML tags for custom styling
                arg = f"<p style='font-size: 18px;'>{arg}</p>"
                kwargs['unsafe_allow_html'] = True
            new_args.append(arg)
        func(*new_args, **kwargs)
    return wrapper

# Apply the decorator to st.write
st.write = custom_write(st.write)


# Sidebar navigation
side_bars = st.sidebar.radio("Navigation", ["Home", "About Us", "Dataset","Coin Correlation","Moving Average", "Visualizations","Predictions","NEWS"])


# Condition for sidebar navigation
if side_bars == "Home":
    home_section()
elif side_bars == "About Us":
    about_us()
    st.info("**Note:** The visualizations and prediction provided here are for informational purposes only and do not constitute financial advice. Always conduct your own research before making investment decisions.")

elif side_bars == "Dataset":
    st.title('Cryptocurrency Dataset')
    st.write("""
    The dataset contains historical price data for 30 various cryptocurrencies, including Bitcoin (BTC), Ethereum (ETH), Ripple (XRP) and etc. 
    Each row represents a specific time period, and the columns provide information about the volume, opening, closing, highest, and lowest prices for each cryptocurrency during that period.
    """)
    st.write("""
    The dataset allows users to analyze the price trends of these cryptocurrencies over time and make informed decisions about their investments. 
    It provides valuable insights into the volatility and performance of the cryptocurrency market.
    """)
    st.subheader("Source:")
    st.write("""
    The dataset was sourced from Yahoo Finance, which aggregates historical cryptocurrency price data from various exchanges and platforms.
    """)
    st.write('Link: https://finance.yahoo.com/topic/stock-market-news')
    # Display a success message
    
    Selection = st.sidebar.radio("Selection",['Dataset','Visualization of Data'])
    if Selection == 'Dataset':
        dataset()
        st.success("Data saved to Cleaned_combined_crypto_data.csv")
    elif Selection == 'Visualization of Data':
        st.title("Visualization of Dataset")
        Vis= st.sidebar.radio("Selection",["Plot Average Price Trend","BoxPlot", 'Volume Volatility', 'Distribution of each coin','Daily Price Change','Crypto Price Volume Movement'])
        
        if Vis == "Plot Average Price Trend":
            st.title('Plot Average Price Trend')
            st.write("Visualize the average price trend of cryptocurrencies over different intervals.")
            # Unique coins
            unique_coins = combined_data['Crypto'].unique()

            # Sidebar - Select cryptocurrency and interval
            selected_coin = st.sidebar.selectbox('Select Cryptocurrency', unique_coins)
            interval = st.sidebar.selectbox('Select Interval', ['Daily', 'Weekly', 'Monthly'])

            # Plot the selected average price trend
            if st.sidebar.button("Plot Average Price Trend"):
                plot_average_price_trend(combined_data, selected_coin, interval)
        elif Vis == "BoxPlot":
            st.title("BoxPlot for each Coin")
            st.write("Visualize the distribution of prices for a specific cryptocurrency using boxplot.")
            # Display coin selection dropdown
            selected_coin = st.sidebar.selectbox("Select a cryptocurrency:", combined_data['Crypto'].unique())

            # Plot boxplot for the selected coin
            if st.sidebar.button("Plot BoxPlot"):
                # plot_boxplot(data, selected_coin)
                plot_boxplot(combined_data, selected_coin)
        elif Vis == 'Volume Volatility':
             plot_crypto_volatility()
        elif Vis == 'Distribution of each coin':
            # Sidebar selection box for cryptocurrencies
            selected_coin = st.sidebar.selectbox('Select a cryptocurrency:', combined_data['Crypto'].unique())

            # Plot distribution and trend line for selected coin
            plot_distribution_and_trend(selected_coin)
        elif Vis == 'Daily Price Change':
            # Sidebar selection box for cryptocurrencies
            selected_coin = st.sidebar.selectbox('Select a cryptocurrency:', combined_data['Crypto'].unique())

            # Plot daily price changes for selected coin
            plot_daily_price_changes(selected_coin)
        elif Vis == 'Crypto Price Volume Movement':
            # Example usage
            visualize_crypto_data(combined_data)
elif side_bars == "Coin Correlation":

    # Pivot the data
    pivoted_data = pivot_data(combined_data)

    # Example usage
    st.title('Coin Correlation Analysis')
    analyze_coin_correlation(pivoted_data)


elif side_bars == "Moving Average":
    st.title("Moving Average Analysis")
    st.write("Interpretation: \n" )
    st.write("1. **Short MA:** Provides insights into short-term price trends. \n 2. **Medium MA:** Offers indications of intermediate-term price movements. \n 3. **Long MA:** Helps identify long-term trends in the cryptocurrency market.")
    plot_moving_average(combined_data)
    

elif side_bars == "Visualizations":
    # Visualization section
    visualization_option = st.sidebar.radio("Select Visualization:", ['Home','Price Comparison', 'Candlestick Chart','Market State Visualization',"Predicted Highs and Lows"])
    if visualization_option == 'Home':
        st.title("Welcome to the World of Data Visualisation for Crypto Data")
        st.image('C:/Users/adedi/OneDrive - Solent University/Semester_2/Soligence_2/Testing/newplot.png', use_container_width=True)
        st.write("Welcome to the Visualization Page! Explore different visualizations to gain insights into cryptocurrency markets.")
        st.write("Main Sections:")
        st.write("- **Price Comparison:** Compare the prices of multiple cryptocurrencies over a specific time period.")
        st.write("- **Candlestick Chart:** Visualize the open, high, low, and close prices of a single cryptocurrency using candlestick charts.")
        st.write("- **Market State Visualization:** Gain insights into the overall market state with visualizations such as market cap distribution, volume analysis, etc.")
        st.write("- **Predicted Highs and Lows:** View predicted high and low prices of cryptocurrencies based on machine learning models.")
        st.write("Choose a visualization from the options above to start exploring!")



    elif visualization_option == 'Price Comparison':
        st.header("Price Comparison of Cryptocurrency")
        st.subheader('**Price Comparison:** ')
        st.write(" Compare the prices of multiple cryptocurrencies over a specific time period.")
        vis_option = st.sidebar.radio("Select how you would like compare the data with Visualization:", ['Metrics', 'Coins'])
        if vis_option == 'Metrics':
            st.header("Price Comparison of Cryptocurrency by Metrics")
            st.write("Compare the prices of multiple cryptocurrencies over a specific time period based on different metrics.")
            coin = st.sidebar.selectbox("Select the cryptocurrency you want to plot:", combined_data['Crypto'].unique())
            metrics = st.sidebar.multiselect("Select price metrics you want to plot (e.g., Close, Open, High, Low):",['Close', 'Open', 'High', 'Low'])
            start_date = st.sidebar.date_input("Enter the start date of the interval:")
            end_date = st.sidebar.date_input("Enter the end date of the interval:")

            if st.sidebar.button("Plot Cryptocurrency Prices", key="plot_button"):
                plot_crypto_metrics(combined_data, coin.upper(), metrics, str(start_date), str(end_date))

        elif vis_option == 'Coins':
            st.header("Price Comparison of Cryptocurrency among Coins")
            st.write("Compare the prices of a single cryptocurrency with other cryptocurrencies over a specific time period.")
            coin = st.sidebar.multiselect("Enter the cryptocurrencies you want to plot ( e.g., BTC-GBP,ETH-USD):",combined_data['Crypto'].unique())
            metric = st.sidebar.selectbox("Enter the price metric you want to plot (e.g., Close, Open, High, Low):",['Close', 'Open', 'High', 'Low'])
            start_date = st.sidebar.date_input("Enter the start date of the interval:")
            end_date = st.sidebar.date_input("Enter the end date of the interval:")

            # Plot the selected cryptocurrency coins for the specified metric within the date range
            if st.sidebar.button("Cryptocurrencies Prices Comparison"):
                plot_crypto_coins(combined_data, coin, metric, str(start_date), str(end_date))


    elif visualization_option == 'Candlestick Chart':
        st.title("Candlestick Chart for Close Price and Volume of Coins")
        crypto_data = pd.read_csv("Cleaned_combined_crypto_data.csv", index_col='Date')
        # User input for coin and period
        available_coins = crypto_data['Crypto'].unique()
        coin = st.selectbox("Select coin", available_coins)
        period_labels = ("Daily", "Weekly", "Monthly")
        period_values = ('D', 'W', 'M')
        period_index = st.radio("Select period", period_labels)
        period = period_values[period_labels.index(period_index)]

        plot_candlestick_chart(coin, period)      
    elif visualization_option == 'Market State Visualization':
        st.title("Market State Visualization for Coins")
        visualize_market_state(combined_data)
        
    elif visualization_option == "Predicted Highs and Lows":
        predicted_highs, predicted_lows = predict_highs_lows(combined_data)
        
elif side_bars == "Predictions":
    prediction_selection = st.sidebar.radio('Selection:', ["Dataset", "Training Model Metrics", "Prediction Graphs","Buy and Sell Prediction","Predict coin by Profit"])
    st.header("Prediction of Cryptocurrency Price")
    if prediction_selection == "Dataset":
        st.subheader("About the Prediction Data")
        st.write("The prediction data is from performing PCA to reduce dimensionality of the data with n_component of 10 and clustering the data with K-means into four clusters and selecting the best from each cluster using the centroid (You can visualize the selected data below)")
        # Display selected data
        # if selected_data is not None:
        if st.button("Display Selected Data"):
            st.dataframe(selected_data)
            # plot_coin_scatter(selected_data)
        else:
            st.write("Click the button above to display the selected data.")
        # else:
        #     st.write("Selected data is not available. Please check the file path and data format.")
        st.write("Here you can make prediction and visualize forecast for the selected coins using one or all of the available models:\n 1. SVR\n 2. XGBoost\n 3. Gradient Boosting\n 4. LSTM")
        display_selected(selected_data)
        
        # Display the scatter plot in Streamlit
        st.title("Coin Scatter Plot")
        plot_coin_scatter(selected_data)
    elif prediction_selection == "Training Model Metrics":
        st.write("### Model Selection:/n Choose a machine learning model from the sidebar and enter the required parameters.")

        st.subheader("Available Models: ")
        st.write("1. **Support Vector Regression (SVR)** \n 2. **Gradient Boosting Regression (GBR)** \n 3. **XGBoost** \n 4. **Long Short-Term Memory (LSTM)**")
        # Load the selected data from a CSV file
        # selected_data = pd.read_csv("Selected_coins.csv", index_col='Date')
        # Streamlit app
        st.title("Model Evaluation for Selected Coins")
        # Select the coins to evaluate
        coins = st.multiselect("Choose the coins you want to evaluate:", selected_data.columns)
        # Select the model
        chosen_model = st.selectbox("Choose the model you want to evaluate:", ['all', 'Gradient Boosting', 'SVR', 'XGBoost', 'LSTM'])
        # Loop through selected coins and evaluate models
        for coin in coins:
            st.subheader(f"Evaluation for {coin}")
            coin_index = selected_data.columns.get_loc(coin)
            evaluate_models_selected_coin(selected_data, coin_index, chosen_model)
    elif prediction_selection == "Prediction Graphs":
        st.title('Cryptocurrency Price Prediction')
        st.write("This page allows you to predict the price of one of the four selected coins using one of the available models. You can choose the frequency of the predictions (daily, weekly, monthly, or quarterly) and specify the number of periods for which you want to make predictions.\n")
        st.write("A table will be displayed showing the predicted prices for the chosen coin over the specified periods.Additionally, a chart will be generated to visualize the predicted prices compared to the actual prices. This chart provides an easy-to-understand visual representation of the accuracy of the predictions.")

        # Select coin for prediction
        selected_coin = st.selectbox('Select a coin for prediction:', selected_data.columns.tolist())

        # User input for selecting the model
        model_choice = st.sidebar.selectbox("Choose the model you want to evaluate:", ['GBR', 'SVR', 'XGB', 'LSTM'])
        # User input for frequency and number of periods
        frequency = st.sidebar.selectbox("Select Frequency", ['daily', 'weekly', 'monthly', 'quarterly'])
        num_periods = st.sidebar.number_input("Enter Number of Periods", min_value=1, value=20)

        # Button to trigger prediction
        if st.button('Load Model and Predict'):
            # Evaluate the model based on the selected coin
            coin_index = selected_data.columns.tolist().index(selected_coin)
            evaluate_and_plot_model(selected_data, coin_index, model_choice)
        else:
            st.write("Click the button above to display the visualization of the prediction of the selected coin.")
    
    elif prediction_selection == "Buy and Sell Prediction":
        
        buy_and_sell = st.sidebar.radio("Selection: ",["Moving Averages","Models"])
        if buy_and_sell == "Moving Averages":
            st.title('Crypto Trading Advisor')
            st.title("Using Moving Average Trading Strategy")
            
            st.write("Here, you can explore predictive models that aim to forecast optimal times to buy and sell assets in financial markets.")
            # Explanation of Buy/Sell Signal
            st.subheader("Explanation of Buy/Sell Signal:")
            st.write("**Buy Signal**: Indicators or conditions suggesting it may be a good time to purchase an asset, "
                     "anticipating its price will increase.")
            st.write("**Sell Signal**: Indicators or conditions suggesting it may be a good time to sell an asset, "
                     "anticipating its price will decrease.")

            # Sidebar for selecting coin and input for number of days

            chosen_coin = st.sidebar.selectbox("Choose the coins you want to evaluate:", selected_data.columns)
            num_days = st.sidebar.number_input("Enter the number of days for forecasting ahead:", min_value=1, step=1)

            # Button to apply MA Trading Strategy
            if st.sidebar.button("Apply MA Trading Strategy"):
                # Call function to apply MA Trading Strategy and display chart
                result_data = determine_best_time_to_trade_future(selected_data, chosen_coin, num_days)

                st.subheader("Buy/Sell Prediction Data")
                st.write(result_data)   
        elif buy_and_sell == "Models":
            # Load data
            selected_data = pd.read_csv("Selected_coins.csv", index_col='Date')

            # Streamlit app
            st.title("Crypto Trading Analysis")

            # Sidebar - Model selection and user input
            model_selection = st.sidebar.radio("Select Model", ("Support Vector Regression (SVR)", "Gradient Boosting Regression (GBR)", "XGBoost", "LSTM"))
            chosen_coin = st.sidebar.selectbox("Choose the coins you want to evaluate:", selected_data.columns)
            num_days = st.sidebar.number_input("Enter the number of days for forecasting:", min_value=1, value=10)
            # Generate lagged features for each coin
            for coin_column in selected_data.columns[:]:
                for lag in range(1, 4):
                    selected_data[f'{coin_column}_lag_{lag}'] = selected_data[coin_column].shift(lag)

            if st.button("Run Analysis"):
                if model_selection == "Support Vector Regression (SVR)":
                    determine_best_time_to_trade(selected_data, chosen_coin, num_days, "SVR")
                elif model_selection == "Gradient Boosting Regression (GBR)":
                    determine_best_time_to_trade(selected_data, chosen_coin, num_days, "GBR")
                elif model_selection == "XGBoost":
                    determine_best_time_to_trade(selected_data, chosen_coin, num_days, "XGBoost")
                elif model_selection == "LSTM":
                    determine_best_time_to_trade(selected_data, chosen_coin, num_days, "LSTM")
          
    elif prediction_selection == "Predict coin by Profit":
        st.title("Profit Prediction Tool")
        st.header("How it Works")
        st.write("""
        1. **Select Model Type**: Choose the type of model you want to use for prediction. You can choose from XGBoost, Gradient Boosting, LSTM, or SVR.

        2. **Enter Parameters**: Input the desired profit amount and the number of days you're willing to wait for the profit.

        3. **Get Results**: The tool will analyze the data and provide you with the closest coin to reach your desired profit within the specified time frame. It will also suggest the next best coin for investment.
        """)
        # selected_data_path = "Selected_coins.csv"
        # Collect user input for model type
        model_type = st.selectbox("Select the model type:", ['Gradient_Boosting', 'SVR', 'Xgboost', 'LSTM'])
        # Collect user input for profit and number of days
        desired_profit = st.number_input("Enter the desired profit amount:")
        num_days = st.number_input("Enter the number of days:", value=1, step=1)
        
        if st.button('Predict Profit'):
            find_best_coins(selected_data)

elif side_bars == 'NEWS':
    st.header("Crypto News")
    st.write("Utilize the search feature to find specific news articles by keywords or topics of interest related to Cryptocurrency. Click on the headline of an article to read the full story.")
    st.subheader("Stay Informed")
    
    crypto = st.text_input("Enter cryptocurrency:", "Bitcoin")
    news_source = st.selectbox("Select news source:", ['all', 'Cryptoslate', 'CoinDesk'])

    # Display top crypto news
    get_top_crypto_news(crypto, news_source=news_source)


