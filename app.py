import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
import requests
import time
import random

# Must be the first Streamlit command
st.set_page_config(page_title="Stock Market Prediction", layout="wide", initial_sidebar_state="expanded")

# Configure yfinance session with proper headers
def setup_yfinance_session():
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
    })
    return session

# Load the trained Keras model
@st.cache_resource
def load_model():
    try:
        return tf.keras.models.load_model('keras_model.h5')
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

model = load_model()

# Enhanced function to fetch historical stock data with multiple fallback methods
@st.cache_data(ttl=300, show_spinner=False)  # Cache for 5 minutes
def fetch_stock_data(stock_symbol, days=120):
    """Fetch stock data with multiple fallback methods"""
    
    def method1_standard():
        """Standard yfinance download"""
        try:
            end_date = datetime.today()
            start_date = end_date - timedelta(days=days)
            df = yf.download(
                stock_symbol, 
                start=start_date, 
                end=end_date,
                progress=False,
                show_errors=False,
                timeout=15,
                threads=True
            )
            if not df.empty and 'Close' in df.columns:
                return df['Close'].values, df.index.to_numpy(), df
        except:
            pass
        return None, None, None

    def method2_ticker_history():
        """Using Ticker.history method"""
        try:
            ticker = yf.Ticker(stock_symbol)
            # Try different period formats
            for period in [f"{days}d", "3mo", "6mo"]:
                try:
                    hist = ticker.history(period=period, timeout=15)
                    if not hist.empty and len(hist) >= 30:
                        return hist['Close'].values, hist.index.to_numpy(), hist
                except:
                    continue
        except:
            pass
        return None, None, None

    def method3_with_session():
        """Using custom session"""
        try:
            session = setup_yfinance_session()
            ticker = yf.Ticker(stock_symbol, session=session)
            hist = ticker.history(period=f"{days}d", timeout=20)
            if not hist.empty:
                return hist['Close'].values, hist.index.to_numpy(), hist
        except:
            pass
        return None, None, None

    def method4_date_range():
        """Using specific date range with retry"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            ticker = yf.Ticker(stock_symbol)
            hist = ticker.history(
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                timeout=20
            )
            if not hist.empty:
                return hist['Close'].values, hist.index.to_numpy(), hist
        except:
            pass
        return None, None, None

    # Try all methods with delays
    methods = [method1_standard, method2_ticker_history, method3_with_session, method4_date_range]
    
    with st.spinner(f"Fetching data for {stock_symbol}..."):
        for i, method in enumerate(methods):
            try:
                # Add random delay to avoid rate limiting
                if i > 0:
                    time.sleep(random.uniform(0.5, 1.5))
                
                result = method()
                if result[0] is not None and len(result[0]) >= 30:
                    st.success(f"✅ Data fetched successfully using method {i+1}")
                    return result
                    
            except Exception as e:
                st.warning(f"Method {i+1} failed: {str(e)[:50]}...")
                continue
    
    return None, None, None

# Enhanced function to fetch stock details with multiple methods
@st.cache_data(ttl=600, show_spinner=False)  # Cache for 10 minutes
def fetch_stock_details(stock_symbol):
    """Fetch stock details with multiple fallback methods"""
    
    def method1_info():
        """Standard info method"""
        try:
            session = setup_yfinance_session()
            stock = yf.Ticker(stock_symbol, session=session)
            time.sleep(random.uniform(0.2, 0.5))
            info = stock.info
            if info and len(info) > 5:
                return info
        except:
            pass
        return None
    
    def method2_no_session():
        """Without custom session"""
        try:
            stock = yf.Ticker(stock_symbol)
            time.sleep(random.uniform(0.2, 0.5))
            info = stock.info
            if info and len(info) > 5:
                return info
        except:
            pass
        return None
    
    def method3_basic_info():
        """Get basic info from history data"""
        try:
            stock = yf.Ticker(stock_symbol)
            hist = stock.history(period="1d")
            if not hist.empty:
                latest = hist.iloc[-1]
                return {
                    'longName': stock_symbol,
                    'currentPrice': float(latest['Close']),
                    'dayHigh': float(latest['High']),
                    'dayLow': float(latest['Low']),
                    'volume': int(latest['Volume']),
                    'regularMarketPrice': float(latest['Close']),
                    'regularMarketDayHigh': float(latest['High']),
                    'regularMarketDayLow': float(latest['Low']),
                    'regularMarketVolume': int(latest['Volume'])
                }
        except:
            pass
        return None
    
    # Try different methods
    methods = [method1_info, method2_no_session, method3_basic_info]
    info = None
    
    for i, method in enumerate(methods):
        try:
            info = method()
            if info:
                break
        except Exception as e:
            continue
    
    # If all methods fail, return defaults
    if not info:
        return stock_symbol.upper(), 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', ''
    
    # Process the info
    currency_symbol = "₹" if ".NS" in stock_symbol or ".BO" in stock_symbol else "$"
    
    # Helper function to format price
    def format_price(price):
        if price and price != 'N/A':
            try:
                return f"{currency_symbol}{float(price):.2f}"
            except:
                return 'N/A'
        return 'N/A'
    
    # Helper function to format volume/market cap
    def format_large_number(num):
        if num and num != 'N/A':
            try:
                num = float(num)
                if num >= 1e12:
                    return f"{num/1e12:.2f}T"
                elif num >= 1e9:
                    return f"{num/1e9:.2f}B"
                elif num >= 1e6:
                    return f"{num/1e6:.2f}M"
                elif num >= 1e3:
                    return f"{num/1e3:.2f}K"
                else:
                    return f"{num:.0f}"
            except:
                return 'N/A'
        return 'N/A'
    
    try:
        company_name = (info.get('longName') or 
                       info.get('shortName') or 
                       stock_symbol.upper())
        
        current_price = format_price(
            info.get('currentPrice') or 
            info.get('regularMarketPrice') or 
            info.get('previousClose')
        )
        
        day_high = format_price(
            info.get('dayHigh') or 
            info.get('regularMarketDayHigh')
        )
        
        day_low = format_price(
            info.get('dayLow') or 
            info.get('regularMarketDayLow')
        )
        
        market_cap = format_large_number(info.get('marketCap'))
        volume = format_large_number(
            info.get('volume') or 
            info.get('regularMarketVolume')
        )
        
        sector = info.get('sector', 'N/A')
        logo_url = info.get('logo_url', '')
        
        return company_name, current_price, day_high, day_low, market_cap, volume, sector, logo_url
        
    except Exception as e:
        return stock_symbol.upper(), 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', ''

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
    if model is None:
        st.error("Model not loaded properly")
        return None, None, None, None
        
    stock_prices, stock_dates, stock_df = fetch_stock_data(stock_symbol)
    if stock_prices is None or len(stock_prices) < 30:
        return None, None, None, None

    input_data, scaler = prepare_model_input(stock_prices)
    predictions = model.predict(input_data).flatten()[:prediction_days]
    predictions = scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
    future_dates = generate_future_dates(stock_dates[-1], len(predictions))

    future_dates = future_dates.astype('datetime64[D]')
    return stock_dates, stock_prices, (future_dates, predictions), stock_df

# Streamlit UI (page config moved to top)

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
        .status-success {
            color: #27ae60;
            font-weight: bold;
        }
        .status-warning {
            color: #f39c12;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="stTitle">🚀 Stock Market Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="stDescription">📊 This model uses Yahoo Finance data to analyze and predict stock behavior for the next few days.</div>', unsafe_allow_html=True)

# Add troubleshooting info
with st.expander("ℹ️ Troubleshooting Tips"):
    st.write("""
    **If you're getting data fetch errors:**
    - Try popular symbols like: AAPL, GOOGL, MSFT, TSLA, AMZN
    - For Indian stocks, use .NS suffix (e.g., RELIANCE.NS, TCS.NS)
    - Wait a few seconds between requests
    - Some symbols might be temporarily unavailable due to Yahoo Finance restrictions
    """)

with st.sidebar:
    st.markdown('[🔍 Find Stock Symbols](https://finance.yahoo.com/lookup)', unsafe_allow_html=True)
    stock_symbol = st.text_input("Stock Symbol (e.g., TSLA, GOOG, MRF.NS for India)", "AAPL")
    
    # Test connection button
    if st.button("🔗 Test Connection"):
        test_data, _, _ = fetch_stock_data(stock_symbol, days=30)
        if test_data is not None:
            st.success("✅ Connection successful!")
        else:
            st.error("❌ Connection failed. Try another symbol.")
    
    # Fetch company details with error handling
    if stock_symbol:
        with st.spinner("Loading stock details..."):
            try:
                company_name, current_price, day_high, day_low, market_cap, volume, sector, logo_url = fetch_stock_details(stock_symbol)
                
                st.write(f"### {company_name}")
                if logo_url:
                    try:
                        st.image(logo_url, width=100)
                    except:
                        pass
                
                # Create a nice info box
                info_col1, info_col2 = st.columns(2)
                with info_col1:
                    st.metric("Current Price", current_price)
                    st.metric("Day High", day_high)
                    st.metric("Market Cap", market_cap)
                
                with info_col2:
                    st.metric("Day Low", day_low)
                    st.metric("Volume", volume)
                    st.write(f"**Sector:** {sector}")
                
                if current_price != 'N/A':
                    st.success("✅ Stock details loaded successfully!")
                else:
                    st.warning("⚠️ Limited data available for this symbol")
                    
            except Exception as e:
                st.error(f"❌ Could not load stock details: {str(e)[:50]}...")
                st.info("💡 Prediction may still work even if details fail to load")
    
    predict_button = st.button("🔮 Predict")

if predict_button:
    if model is None:
        st.error("❌ Model not loaded. Please check if 'keras_model.h5' exists.")
    else:
        with st.spinner("Processing prediction..."):
            stock_dates, historical_prices, prediction_data, stock_df = predict_stock_prices(stock_symbol)

        if historical_prices is None:
            st.error("❌ Could not fetch stock data. Please try:")
            st.write("1. A different stock symbol (e.g., AAPL, GOOGL, MSFT)")
            st.write("2. Adding .NS for Indian stocks (e.g., RELIANCE.NS)")
            st.write("3. Waiting a few minutes and trying again")
            st.write("4. Check if the symbol exists on Yahoo Finance")
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
            ax.plot(stock_dates, historical_prices, label='📈 Historical Prices', color='#3498db', linewidth=2)
            ax.plot(extended_dates, extended_prices, linestyle='dashed', color='#e74c3c', linewidth=2, label='📊 Future Trend')
            ax.scatter(future_dates, predictions, color='#2ecc71', marker='o', label='🎯 Predicted Prices', s=60)
            ax.set_xlim(stock_dates[0], future_dates[-1] + np.timedelta64(10, 'D'))
            ax.set_xticks(stock_dates[::10])
            plt.xticks(rotation=45, fontsize=12, color='#2c3e50')
            plt.yticks(fontsize=12, color='#2c3e50')
            ax.set_title(f"Stock Price Prediction for {stock_symbol}", fontsize=16, color='#2c3e50')
            ax.set_xlabel("Date", fontsize=14, color='#2c3e50')
            ax.set_ylabel("Stock Price", fontsize=14, color='#2c3e50')
            ax.legend()
            st.pyplot(fig)
            
            st.subheader(f"📋 Stock Details for {stock_symbol} (Previous 5 Days)")
            if stock_df is not None and not stock_df.empty:
                st.write(stock_df.tail(5))
            
            st.subheader("💡 Investment Insights")
            st.write("Based on the predicted trend, this stock may show an upward or downward movement.")
            st.write("It's advisable to analyze market conditions and company performance before investing.")
            st.write(f"For real-time news and updates, check [latest stock news](https://www.google.com/search?q={stock_symbol}+stock+news).")
            st.markdown('<div class="stFooter">Made by <a href="https://www.linkedin.com/in/nabhya-sharma-b0a374248/" target="_blank">Nabhya Sharma</a></div>', unsafe_allow_html=True)
