import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from alpha_vantage.timeseries import TimeSeries
import time

# API Key (Store securely, e.g., as an environment variable in production)
API_KEY = ""  # Replace with your Alpha Vantage API key

# Function to fetch real-time current price
def fetch_current_price(ticker):
    ts = TimeSeries(key=API_KEY, output_format='pandas')
    try:
        if ticker.upper() in ["BTC", "ETH", "XRP"]:  # Cryptocurrency check
            data, meta_data = ts.get_digital_currency_exchange_rate(
                from_currency=ticker.upper(),
                to_currency="USD"
            )
            current_price = float(data['5. Exchange Rate'])
            return current_price
        else:  # Traditional stocks
            # Use the latest price from daily data as an approximation
            data, meta_data = ts.get_daily(symbol=ticker, outputsize='compact')
            data = data.rename(columns={
                '1. open': 'Open',
                '2. high': 'High',
                '3. low': 'Low',
                '4. close': 'Close',
                '5. volume': 'Volume'
            })
            current_price = float(data['Close'][-1])  # Latest available price
            return current_price
    except AttributeError:
        st.warning("Real-time exchange rate method not available. Using latest historical price as fallback.")
        # Fallback: Use the latest price from historical data
        historical_data = fetch_stock_data(ticker, days_back=1)  # Fetch only 1 day of data
        return float(historical_data['Close'][-1])
    except Exception as e:
        raise Exception(f"Failed to fetch current price: {str(e)}")

# Function to fetch historical stock data using Alpha Vantage with rate limit handling
def fetch_stock_data(ticker, days_back=365):
    ts = TimeSeries(key=API_KEY, output_format='pandas')
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            if ticker.upper() in ["BTC", "ETH", "XRP"]:  # Cryptocurrency check
                data, meta_data = ts.get_digital_currency_daily(
                    symbol=ticker.upper(),
                    market="USD"
                )
                data = data.rename(columns={
                    '1a. open (USD)': 'Open',
                    '2a. high (USD)': 'High',
                    '3a. low (USD)': 'Low',
                    '4a. close (USD)': 'Close',
                    '5. volume': 'Volume'
                })
            else:  # Traditional stocks
                data, meta_data = ts.get_daily(symbol=ticker, outputsize='full')
                data = data.rename(columns={
                    '1. open': 'Open',
                    '2. high': 'High',
                    '3. low': 'Low',
                    '4. close': 'Close',
                    '5. volume': 'Volume'
                })
            
            data = data[(data.index >= start_date) & (data.index <= end_date)]
            if data.empty:
                raise ValueError(f"No data available for ticker {ticker} from {start_date} to {end_date}")
            st.write("Raw Historical Data Sample:", data.head())  # Debugging output
            time.sleep(12)  # Respect Alpha Vantage's 5 requests per minute limit
            return data
        except Exception as e:
            if "rate" in str(e).lower() or "limit" in str(e).lower():
                wait_time = 60  # Wait 1 minute before retrying
                st.warning(f"Rate limit hit. Retrying in {wait_time} seconds... (Attempt {attempt + 1}/{max_retries})")
                time.sleep(wait_time)
            else:
                raise
    raise Exception("Failed to fetch data after maximum retries.")

# Function to prepare data for LSTM
def prepare_data(data, look_back=60):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))
    
    X, y = [], []
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i - look_back:i, 0])
        y.append(scaled_data[i, 0])
    X, y = np.array(X), np.array(y)
    
    # Reshape for LSTM [samples, time steps, features]
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    return X, y, scaler

# Function to build and train LSTM model
def build_lstm_model(X_train, y_train):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)
    return model

# Function to predict future prices
def predict_future_prices(model, last_sequence, scaler, days_to_predict=5):
    future_predictions = []
    current_sequence = last_sequence.copy()
    
    for _ in range(days_to_predict):
        next_pred = model.predict(current_sequence.reshape(1, current_sequence.shape[0], 1))
        future_predictions.append(next_pred[0, 0])
        current_sequence = np.roll(current_sequence, -1)
        current_sequence[-1] = next_pred[0, 0]
    
    future_predictions = np.array(future_predictions).reshape(-1, 1)
    future_predictions = scaler.inverse_transform(future_predictions)
    return future_predictions

# Streamlit frontend
def main():
    st.title("Real-Time Stock Price Prediction")
    st.write("Enter a stock ticker to fetch real-time data and predict future prices.")

    # User input
    ticker = st.text_input("Stock Ticker (e.g., TSLA, AAPL):", "TSLA").upper()
    days_back = st.slider("Days of Historical Data:", 100, 1000, 365)
    days_to_predict = st.slider("Days to Predict:", 1, 10, 5)

    if st.button("Predict"):
        with st.spinner("Fetching data and training model..."):
            try:
                # Fetch real-time current price
                current_price = fetch_current_price(ticker)
                st.write(f"Current Price of {ticker}: ${current_price:.2f}")

                # Fetch historical data for training
                stock_data = fetch_stock_data(ticker, days_back)
                if stock_data.empty:
                    st.error("Invalid ticker or no historical data available.")
                    return

                # Prepare data
                look_back = 60
                X, y, scaler = prepare_data(stock_data, look_back)
                train_size = int(len(X) * 0.8)
                X_train, X_test = X[:train_size], X[train_size:]
                y_train, y_test = y[:train_size], y[train_size:]

                # Train model
                model = build_lstm_model(X_train, y_train)

                # Predict on test data
                predictions = model.predict(X_test)
                predictions = scaler.inverse_transform(predictions)
                y_test_scaled = scaler.inverse_transform([y_test])

                # Predict future prices
                last_sequence = X[-1]
                future_predictions = predict_future_prices(model, last_sequence, scaler, days_to_predict)

                # Plot historical vs predicted
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(stock_data.index[-len(y_test):], y_test_scaled.T, label="Actual Price")
                ax.plot(stock_data.index[-len(predictions):], predictions, label="Predicted Price")
                ax.set_title(f"{ticker} Stock Price Prediction")
                ax.set_xlabel("Date")
                ax.set_ylabel("Price (USD)")
                ax.legend()
                st.pyplot(fig)

                # Plot future predictions
                future_dates = [stock_data.index[-1] + timedelta(days=i+1) for i in range(days_to_predict)]
                fig2, ax2 = plt.subplots(figsize=(12, 6))
                ax2.plot(future_dates, future_predictions, label="Future Predictions", marker='o')
                ax2.set_title(f"{ticker} Future Price Prediction ({days_to_predict} Days)")
                ax2.set_xlabel("Date")
                ax2.set_ylabel("Price (USD)")
                ax2.legend()
                st.pyplot(fig2)

                # Display future predictions
                st.write("Future Predictions:")
                for date, price in zip(future_dates, future_predictions):
                    st.write(f"{date.strftime('%Y-%m-%d')}: ${price[0]:.2f}")

            except ValueError as e:
                st.error(str(e))
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()