import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
import requests
import time
import warnings
warnings.filterwarnings('ignore')

# Must be the first Streamlit command
st.set_page_config(page_title="Stock Market Prediction", layout="wide", initial_sidebar_state="expanded")

# Load the trained Keras model
@st.cache_resource
def load_model():
    try:
        return tf.keras.models.load_model('keras_model.h5')
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

model = load_model()

# Alpha Vantage API Key - Replace with your actual API key
ALPHA_VANTAGE_API_KEY = "XL2JSQ1Q4PBP0AE2"  # Replace this with your actual API key

# Alpha Vantage data fetching function
def fetch_from_alpha_vantage(stock_symbol, api_key=ALPHA_VANTAGE_API_KEY):
    """Fetch data from Alpha Vantage API"""
    try:
        url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={stock_symbol}&apikey={api_key}"
        response = requests.get(url, timeout=30)
        data = response.json()
        
        # Check for API error messages
        if "Error Message" in data:
            return None, None, None, f"Error: {data['Error Message']}"
        
        if "Note" in data:
            return None, None, None, "API call frequency limit reached. Please try again later."
        
        if "Information" in data:
            return None, None, None, f"API Info: {data['Information']}"
        
        if "Time Series (Daily)" in data:
            time_series = data["Time Series (Daily)"]
            dates = []
            closes = []
            highs = []
            lows = []
            volumes = []
            
            # Sort dates to ensure chronological order
            sorted_dates = sorted(time_series.keys())
            
            for date_str in sorted_dates:
                values = time_series[date_str]
                dates.append(np.datetime64(date_str))
                closes.append(float(values["4. close"]))
                highs.append(float(values["2. high"]))
                lows.append(float(values["3. low"]))
                volumes.append(int(values["5. volume"]))
            
            if len(closes) >= 10:
                df = pd.DataFrame({
                    'Date': dates,
                    'Close': closes,
                    'High': highs,
                    'Low': lows,
                    'Volume': volumes
                })
                return np.array(closes), np.array(dates), df, "Success"
        
        return None, None, None, "No data found in API response"
        
    except requests.exceptions.Timeout:
        return None, None, None, "Request timeout. Please try again."
    except requests.exceptions.RequestException as e:
        return None, None, None, f"Network error: {str(e)}"
    except Exception as e:
        return None, None, None, f"Unexpected error: {str(e)}"

# Main data fetching function
@st.cache_data(ttl=300, show_spinner=False)
def fetch_stock_data(stock_symbol):
    """Fetch stock data from Alpha Vantage"""
    
    stock_symbol = stock_symbol.strip().upper()
    
    with st.spinner(f"Fetching data for {stock_symbol} from Alpha Vantage..."):
        prices, dates, df, message = fetch_from_alpha_vantage(stock_symbol)
        
        if prices is not None and len(prices) >= 10:
            st.success(f"✅ Data fetched successfully from Alpha Vantage")
            st.write(f"📊 Retrieved {len(prices)} data points")
            return prices, dates, df
        else:
            st.error(f"❌ Alpha Vantage API Error: {message}")
            return None, None, None

# Enhanced stock details function using Alpha Vantage
@st.cache_data(ttl=600, show_spinner=False)
def fetch_stock_details(stock_symbol):
    """Fetch stock details from Alpha Vantage"""
    
    stock_symbol = stock_symbol.strip().upper()
    
    try:
        # Fetch overview data
        overview_url = f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={stock_symbol}&apikey={ALPHA_VANTAGE_API_KEY}"
        overview_response = requests.get(overview_url, timeout=30)
        overview_data = overview_response.json()
        
        # Fetch quote data for current price
        quote_url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={stock_symbol}&apikey={ALPHA_VANTAGE_API_KEY}"
        quote_response = requests.get(quote_url, timeout=30)
        quote_data = quote_response.json()
        
        # Process overview data
        company_name = overview_data.get('Name', stock_symbol.upper())
        market_cap = overview_data.get('MarketCapitalization', 'N/A')
        sector = overview_data.get('Sector', 'N/A')
        
        # Process quote data
        quote_info = quote_data.get('Global Quote', {})
        current_price = quote_info.get('05. price', 'N/A')
        day_high = quote_info.get('03. high', 'N/A')
        day_low = quote_info.get('04. low', 'N/A')
        volume = quote_info.get('06. volume', 'N/A')
        
        # Format data
        def format_price(price):
            if price and price != 'N/A':
                try:
                    return f"${float(price):.2f}"
                except:
                    return 'N/A'
            return 'N/A'
        
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
        
        current_price_formatted = format_price(current_price)
        day_high_formatted = format_price(day_high)
        day_low_formatted = format_price(day_low)
        market_cap_formatted = format_large_number(market_cap)
        volume_formatted = format_large_number(volume)
        
        return company_name, current_price_formatted, day_high_formatted, day_low_formatted, market_cap_formatted, volume_formatted, sector
        
    except Exception as e:
        return stock_symbol.upper(), 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A'

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
    # Ensure we have enough data
    if len(stock_prices) < 30:
        # If we don't have enough data, pad with the last value
        padding_needed = 30 - len(stock_prices)
        last_price = stock_prices[-1]
        padded_prices = np.concatenate([np.full(padding_needed, last_price), stock_prices])
        scaled_prices, scaler = normalize_data(padded_prices)
    else:
        scaled_prices, scaler = normalize_data(stock_prices)
    
    return np.array(scaled_prices[-30:]).reshape(1, 30, 1), scaler

# Function to predict stock prices
def predict_stock_prices(stock_symbol, prediction_days=30):
    if model is None:
        st.error("Model not loaded properly")
        return None, None, None, None
        
    stock_prices, stock_dates, stock_df = fetch_stock_data(stock_symbol)
    if stock_prices is None or len(stock_prices) < 5:
        return None, None, None, None

    try:
        input_data, scaler = prepare_model_input(stock_prices)
        predictions = model.predict(input_data).flatten()[:prediction_days]
        predictions = scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
        future_dates = generate_future_dates(stock_dates[-1], len(predictions))

        future_dates = future_dates.astype('datetime64[D]')
        return stock_dates, stock_prices, (future_dates, predictions), stock_df
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None, None, None, None

# Streamlit UI

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
        .stock-link {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 12px 20px;
            border-radius: 8px;
            text-decoration: none;
            font-weight: bold;
            display: inline-block;
            margin: 10px 0;
            transition: transform 0.2s;
        }
        .stock-link:hover {
            transform: translateY(-2px);
            text-decoration: none;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="stTitle">🚀 Stock Market Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="stDescription">📊 This model uses Alpha Vantage API to analyze and predict stock behavior for the next few days.</div>', unsafe_allow_html=True)

# Add Alpha Vantage info
with st.expander("📡 Alpha Vantage Information"):
    st.write("""
    **This app uses Alpha Vantage API for stock data:**
    
    - **Real-time and historical stock data**
    - **Company fundamentals and overview**
    - **Global market coverage**
    
    **Popular Stock Symbols to Try:**
    - **US Stocks:** AAPL, GOOGL, MSFT, TSLA, AMZN, NVDA, META
    - **International:** Use appropriate exchange suffixes
    
    **API Limitations:**
    - Free tier: 5 API requests per minute, 500 requests per day
    - Premium: Higher limits available
    
    **Get your free API key:** [Alpha Vantage API Key](https://www.alphavantage.co/support/#api-key)
    """)

with st.sidebar:
    st.markdown('### 📈 Stock Selection')
    stock_symbol = st.text_input("Stock Symbol (e.g., AAPL, GOOGL, MSFT)", "AAPL")
    
    # Fetch company details with error handling
    if stock_symbol:
        with st.spinner("Loading stock details..."):
            try:
                company_name, current_price, day_high, day_low, market_cap, volume, sector = fetch_stock_details(stock_symbol)
                
                st.write(f"### {company_name}")
                
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
                    st.success("✅ Stock details loaded!")
                else:
                    st.warning("⚠️ Limited data available")
                    
            except Exception as e:
                st.error(f"❌ Could not load stock details")
                st.info("💡 Prediction may still work")
    
    predict_button = st.button("Predict")
    
    # Browse all available stocks link
    st.markdown("""
    <style>
        .stock-link {
            display: inline-block;
            padding: 12px 20px;
            border-radius: 12px;
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white !important;
            text-decoration: none !important;
            font-weight: bold;
            font-size: 16px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        .stock-link:hover {
            transform: scale(1.03);
            box-shadow: 0 6px 10px rgba(0, 0, 0, 0.15);
        }
    </style>

    <div style="text-align: center; margin: 30px 0;">
        <a href="https://www.alphavantage.co/query?function=LISTING_STATUS&apikey=demo" 
           target="_blank" 
           class="stock-link">
            Alpha Vantage Stock Search Guide
        </a>
    </div>
""", unsafe_allow_html=True)



if predict_button:
    if model is None:
        st.error("❌ Model not loaded. Please check if 'keras_model.h5' exists.")
    else:
        with st.spinner("Processing prediction using Alpha Vantage data..."):
            stock_dates, historical_prices, prediction_data, stock_df = predict_stock_prices(stock_symbol)

        if historical_prices is None:
            st.error("❌ Could not fetch stock data from Alpha Vantage. Please check:")
            st.write("1. Your API key is valid and has remaining quota")
            st.write("2. The stock symbol is correct (e.g., AAPL, GOOGL, MSFT)")
            st.write("3. Your internet connection")
            st.write("4. Try again after a minute (API rate limits)")
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
            ax.set_xticks(stock_dates[::max(1, len(stock_dates)//10)])
            plt.xticks(rotation=45, fontsize=12, color='#2c3e50')
            plt.yticks(fontsize=12, color='#2c3e50')
            ax.set_title(f"Stock Price Prediction for {stock_symbol}", fontsize=16, color='#2c3e50')
            ax.set_xlabel("Date", fontsize=14, color='#2c3e50')
            ax.set_ylabel("Stock Price", fontsize=14, color='#2c3e50')
            ax.legend()
            st.pyplot(fig)
            
            st.subheader(f"📋 Stock Details for {stock_symbol} (Last Available Days)")
            if stock_df is not None and not stock_df.empty:
                display_df = stock_df.tail(min(5, len(stock_df)))
                st.write(display_df)
            
            st.subheader("💡 Investment Insights")
            current_price = historical_prices[-1]
            avg_prediction = np.mean(predictions)
            trend = "upward" if avg_prediction > current_price else "downward"
            
            st.write(f"📊 **Current Price:** ${current_price:.2f}")
            st.write(f"📈 **Average Predicted Price:** ${avg_prediction:.2f}")
            st.write(f"📉 **Predicted Trend:** {trend.capitalize()}")
            st.write("⚠️ **Disclaimer:** This prediction is based on historical data and should not be considered as financial advice.")
            st.write("🔍 It's advisable to analyze market conditions and company performance before investing.")
            st.write(f"📰 For real-time news and updates, check [latest stock news](https://www.google.com/search?q={stock_symbol}+stock+news).")
            st.markdown('<div class="stFooter">Made by <a href="https://www.linkedin.com/in/nabhya-sharma-b0a374248/" target="_blank">Nabhya Sharma</a></div>', unsafe_allow_html=True)
