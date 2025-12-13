# Real-Time Stock Trend Forecaster (LSTM)

## Overview
This project is a Time Series forecasting model using Long Short-Term Memory (LSTM) neural networks to predict stock market trends. It integrates real-time data pipelines from Yahoo Finance and utilizes technical indicators like RSI and MACD for enhanced predictive accuracy.

## Features
- **Real-time Data**: Fetches live stock data using Yahoo Finance API.
- **Technical Indicators**: Calculates RSI (Relative Strength Index) and MACD (Moving Average Convergence Divergence).
- **Deep Learning**: Uses LSTM (Long Short-Term Memory) networks for time-series prediction.
- **Visualization**: Plots historical data, indicators, and predictions.

## Tech Stack
- Python
- TensorFlow/Keras
- Pandas, NumPy
- Yahoo Finance API (`yfinance`)

## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the training script:
   ```bash
   python src/train.py
   ```
3. Run the dashboard (optional/future work):
   ```bash
   python app.py
   ```
