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

# Build and train the LSTM model
model = Sequential()
model.add(LSTM(64, return_sequences=True, input_shape=(7, 1)))
model.add(LSTM(32))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X, y, epochs=10, batch_size=32)

# Make predictions for the next 7 days
predictions = []
for ticker in tickers:
  prices = data[ticker]["Close"].values
  last_prices = prices[-7:]
  prediction = model.predict(np.array([last_prices]).reshape(1, 7, 1))
  predictions.append(prediction[0][0])

# Compare predictions with actual prices
for i, ticker in enumerate(tickers):
  real_price = data[ticker]["Close"].iloc[-1]
  predicted_price = predictions[i]
  print(f"Ticker: {ticker}")
  print(f"Real Price: {real_price}")
  print(f"Predicted Price: {predicted_price}")
  print(f"Difference: {abs(real_price - predicted_price)}")

# Generate visualizations for each stock
for i, ticker in enumerate(tickers):
  real_price = data[ticker]["Close"].iloc[-1]
  predicted_price = predictions[i]
  
  plt.figure(figsize=(10, 6))
  plt.plot(data[ticker].index, data[ticker]["Close"], label="Historical Prices")
  plt.plot(data[ticker].index[-7:], [real_price] * 7, marker='o', label="Actual Price")
  plt.plot(data[ticker].index[-7:], [predicted_price] * 7, marker='x', label="Predicted Price")
  plt.title(f"{ticker} - Stock Prices")
  plt.xlabel("Date")
  plt.ylabel("Price")
  plt.legend()
  plt.grid(True)
  plt.show()

