import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
import yfinance as yf

# Define the list of ticker symbols
tickers = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'F', 'ADBE', 'TSLA', 'JNJ', 'WBD', 'KO']

# Download historical stock data for each ticker
data = {}
for ticker in tickers:
    data[ticker] = yf.download(ticker, start="2018-01-01", end="2022-12-22")

# Convert dataframes to numpy arrays for model training
X = []
y = []
for ticker in tickers:
    prices = data[ticker]["Close"].values
    X_tmp = []
    y_tmp = []
    for i in range(len(prices) - 7):
        X_tmp.append(prices[i:i+7])
        y_tmp.append(prices[i+7])
    X.extend(X_tmp)
    y.extend(y_tmp)

X = np.array(X)
y = np.array(y)

# Reshape X to have the shape (samples, time_steps, features)
X = X.reshape(X.shape[0], X.shape[1], 1)

# Build and train the LSTM model
model = Sequential()
model.add(LSTM(64, return_sequences=True, input_shape=(7, 1)))
model.add(LSTM(32))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X, y, epochs=10, batch_size=32)  # Increase epochs for better training

# Make step-by-step predictions for the next 7 days
predictions = {}
for ticker in tickers:
    prices = data[ticker]["Close"].values
    last_prices = prices[-7:].tolist()
    predicted_prices = []
    for _ in range(7):
        prediction = model.predict(np.array([last_prices[-7:]]).reshape(1, 7, 1))
        predicted_price = prediction[0][0]
        predicted_prices.append(predicted_price)
        last_prices.append(predicted_price)
    predictions[ticker] = predicted_prices

# Download actual stock data for the next 7 days to compare predictions
actual_data = {}
for ticker in tickers:
    actual_data[ticker] = yf.download(ticker, start="2022-12-23", end="2022-12-30")

# Generate visualizations and comparisons for each stock
for ticker in tickers:
    prices = data[ticker]["Close"].values
    real_prices = actual_data[ticker]["Close"].values
    predicted_prices = predictions[ticker]

    # Print actual and predicted prices to debug
    print(f"Ticker: {ticker}")
    print(f"Actual Prices: {real_prices}")
    print(f"Predicted Prices: {predicted_prices}")

    # Check if we have enough actual prices
    if len(real_prices) == 7:
        differences = real_prices - predicted_prices

        plt.figure(figsize=(10, 6))
        plt.plot(pd.date_range(start="2022-12-23", periods=7), real_prices, marker='o', label="Actual Prices")
        plt.plot(pd.date_range(start="2022-12-23", periods=7), predicted_prices, marker='x', label="Predicted Prices")
        plt.title(f"{ticker} - Actual vs Predicted Stock Prices")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.grid(True)
        plt.show()

        # Print the differences
        print(f"Differences: {differences}")
    else:
        print(f"Not enough actual data for {ticker} to compare predictions (only {len(real_prices)} actual prices available).")
