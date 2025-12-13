# Real-Time Stock Trend Forecaster (LSTM)

## 🚀 Live Demo
**[Deploy to Streamlit Cloud](https://streamlit.io/cloud)**  
*Note: GitHub Pages only hosts static websites (HTML/CSS). Since this is a Python application requiring a server to run TensorFlow/Pandas, you must deploy it to a platform like Streamlit Cloud, Render, or Railway.*

## Overview
This project is a Time Series forecasting model using Long Short-Term Memory (LSTM) neural networks to predict stock market trends. It integrates real-time data pipelines from Yahoo Finance and utilizes technical indicators like RSI and MACD for enhanced predictive accuracy.

## Features
- **Real-time Data**: Fetches live stock data using Yahoo Finance API.
- **Technical Indicators**: Calculates RSI (Relative Strength Index) and MACD (Moving Average Convergence Divergence).
- **Deep Learning**: Uses LSTM (Long Short-Term Memory) networks for time-series prediction.
- **Interactive Dashboard**: Streamlit-based web interface for easy interaction.
- **Visualization**: Plots historical data, indicators, and predictions using Plotly.

## Tech Stack
- **Frontend/UI**: Streamlit
- **Core**: Python
- **ML/DL**: TensorFlow/Keras, Scikit-learn
- **Data**: Pandas, NumPy, Yahoo Finance API (`yfinance`)
- **Visualization**: Plotly

## Setup & Running Locally

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Interactive Dashboard (Recommended)**:
   This will launch the web application in your browser.
   ```bash
   streamlit run app.py
   ```

3. **Run the CLI Training Script**:
   To train the model via command line without the web UI:
   ```bash
   python run.py
   ```

## Deployment Guide (Streamlit Cloud)
1. Push this code to a GitHub repository.
2. Go to [share.streamlit.io](https://share.streamlit.io/).
3. Connect your GitHub account.
4. Select this repository and the `app.py` file.
5. Click **Deploy**!

