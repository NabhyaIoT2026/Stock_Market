import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler

# Load the trained Keras model
model = tf.keras.models.load_model('keras_model.h5')

# Function to fetch historical stock data
def fetch_stock_data(stock_symbol, days=120):
    end_date = datetime.today()
    start_date = end_date - timedelta(days=days)
    df = yf.download(stock_symbol, start=start_date, end=end_date)
    return df['Close'].values if 'Close' in df else None, df.index.to_numpy() if 'Close' in df else None, df

# Function to fetch stock details
def fetch_stock_details(stock_symbol):
    try:
        st.markdown('[🔍 Find Stock Symbols](https://finance.yahoo.com/lookup)', unsafe_allow_html=True)  # Added link
        stock = yf.Ticker(stock_symbol)
        info = stock.info
        currency_symbol = "₹" if ".NS" in stock_symbol or ".BO" in stock_symbol else "$"
        
        return (info.get('longName', 'Unknown Company'),
                f"{currency_symbol}{info.get('currentPrice', 'N/A')}" if 'currentPrice' in info else 'N/A',
                f"{currency_symbol}{info.get('dayHigh', 'N/A')}" if 'dayHigh' in info else 'N/A',
                f"{currency_symbol}{info.get('dayLow', 'N/A')}" if 'dayLow' in info else 'N/A',
                info.get('marketCap', 'N/A'),
                info.get('volume', 'N/A'),
                info.get('sector', 'N/A'),
                info.get('logo_url', ''))
    except Exception as e:
        return 'Unknown Company', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', ''

# Function to generate future dates
def generate_future_dates(start_date, days):
    return np.array([start_date + np.timedelta64(i, 'D') for i in range(1, days + 1)], dtype='datetime64[D]')

# Function to normalize data
def normalize_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data.reshape(-1, 1))
    return scaled_data, scaler

# Function to prepare model input
def prepare_model_input(stock_prices):
    scaled_prices, scaler = normalize_data(stock_prices)
    return np.array(scaled_prices[-30:]).reshape(1, 30, 1), scaler

# Function to predict stock prices
def predict_stock_prices(stock_symbol, prediction_days=30):
    stock_prices, stock_dates, stock_df = fetch_stock_data(stock_symbol)
    if stock_prices is None or len(stock_prices) < 30:
        return None, None, None, None

    input_data, scaler = prepare_model_input(stock_prices)
    predictions = model.predict(input_data).flatten()[:prediction_days]
    predictions = scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
    future_dates = generate_future_dates(stock_dates[-1], len(predictions))

    future_dates = future_dates.astype('datetime64[D]')
    return stock_dates, stock_prices, (future_dates, predictions), stock_df

# Streamlit UI with left-aligned input
st.set_page_config(page_title="Stock Market Prediction", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
        .stApp {
            padding: 20px;
            background-color: #CAF1DE;
        }
        .stTitle {
            text-align: center;
            font-size: 36px;
            font-weight: bold;
            color: #2c3e50;
            padding: 15px;
            background: #E1F8DC;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .stDescription {
            text-align: center;
            font-size: 18px;
            font-weight: bold;
            color: #34495e;
            background: #F9E79F;
            padding: 10px;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0 3px 5px rgba(0, 0, 0, 0.1);
        }
        header { 
            background-color: #CAF1DE !important;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="stTitle">🚀 Stock Market Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="stDescription">📊 This model uses Yahoo Finance data to analyze and predict stock behavior for the next few days.</div>', unsafe_allow_html=True)

with st.sidebar:
    stock_symbol = st.text_input("Stock Symbol (e.g., TSLA, GOOG, MRF.NS for India)", "TSLA")
    company_name, current_price, day_high, day_low, market_cap, volume, sector, logo_url = fetch_stock_details(stock_symbol)
    st.write(f"### {company_name}")
    if logo_url:
        st.image(logo_url, width=100)
    st.write(f"**Current Price:** {current_price}")
    st.write(f"**Day High:** {day_high}")
    st.write(f"**Day Low:** {day_low}")
    st.write(f"**Market Cap:** {market_cap}")
    st.write(f"**Volume:** {volume}")
    st.write(f"**Sector:** {sector}")
    st.write("This information provides insights into the stock's current performance, including market trends and industry relevance.")
    predict_button = st.button("🔮 Predict")

if predict_button:
    stock_dates, historical_prices, prediction_data, stock_df = predict_stock_prices(stock_symbol)

    if historical_prices is None:
        st.error("❌ Could not fetch stock data. Please try another symbol.")
    else:
        future_dates, predictions = prediction_data
        predictions = np.array(predictions).flatten()
        future_dates = np.array(future_dates, dtype='datetime64[D]')

        last_date = np.array([stock_dates[-1]], dtype='datetime64[D]')
        last_price = np.array([historical_prices[-1]]).reshape(-1, 1)
        predictions = np.array(predictions).reshape(-1, 1)

        extended_dates = np.concatenate((last_date, future_dates))
        extended_prices = np.concatenate((last_price, predictions)).flatten()

        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(stock_dates, historical_prices, label=' Historical Prices', color='#3498db', linewidth=2)
        ax.plot(extended_dates, extended_prices, linestyle='dashed', color='#e74c3c', linewidth=2, label=' Future Trend')
        ax.scatter(future_dates, predictions, color='#2ecc71', marker='o', label=' Predicted Prices', s=60)
        ax.set_xlim(stock_dates[0], future_dates[-1] + np.timedelta64(10, 'D'))
        ax.set_xticks(stock_dates[::10])
        plt.xticks(rotation=45, fontsize=12, color='#2c3e50')
        plt.yticks(fontsize=12, color='#2c3e50')
        ax.set_title(f"Stock Price Prediction for {stock_symbol}", fontsize=16, color='#2c3e50')
        ax.set_xlabel("Date", fontsize=14, color='#2c3e50')
        ax.set_ylabel("Stock Price", fontsize=14, color='#2c3e50')
        ax.legend()
        st.pyplot(fig)
        
        st.subheader(f"Stock Details for {stock_symbol} (Previous 5 Days)")
        st.write(stock_df.tail(5))
        st.subheader("Investment Insights")
        st.write("Based on the predicted trend, this stock may show an upward or downward movement.")
        st.write("It's advisable to analyze market conditions and company performance before investing.")
        st.write(f"For real-time news and updates, check [latest stock news](https://www.google.com/search?q={stock_symbol}+stock+news).")
        st.markdown('<div class="stFooter">Made by <a href="https://www.linkedin.com/in/nabhya-sharma-b0a374248/" target="_blank">Nabhya Sharma</a></div>', unsafe_allow_html=True)
