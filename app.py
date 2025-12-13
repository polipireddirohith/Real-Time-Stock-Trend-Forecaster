import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from src.data_processor import DataProcessor
from src.model import build_lstm_model
import tensorflow as tf

st.set_page_config(page_title="Real-Time Stock Forecaster", layout="wide")

st.title("📈 Real-Time Stock Trend Forecaster (LSTM)")
st.markdown("""
This application uses **Long Short-Term Memory (LSTM)** neural networks to predict stock price trends.
It fetches real-time data from Yahoo Finance, calculates technical indicators (RSI, MACD), and forecasts future prices.
""")

# Sidebar
st.sidebar.header("Configuration")
ticker = st.sidebar.text_input("Stock Ticker", value="AAPL")
epochs = st.sidebar.slider("Training Epochs", min_value=1, max_value=50, value=5)
batch_size = st.sidebar.slider("Batch Size", min_value=16, max_value=64, value=32)

if st.sidebar.button("Run Forecast"):
    status_text = st.empty()
    progress_bar = st.progress(0)
    
    try:
        # 1. Fetch Data
        status_text.text("Fetching real-time data...")
        processor = DataProcessor(ticker)
        processor.fetch_data()
        progress_bar.progress(20)
        
        # 2. Add Indicators
        status_text.text("Calculating technical indicators (RSI, MACD)...")
        processor.add_technical_indicators()
        progress_bar.progress(40)
        
        # Display Raw Data
        st.subheader(f"Raw Data for {ticker}")
        st.dataframe(processor.data.tail())
        
        # Plot Indicators
        fig_ind = go.Figure()
        fig_ind.add_trace(go.Scatter(x=processor.data['Date'], y=processor.data['Close'], name='Close Price'))
        fig_ind.add_trace(go.Scatter(x=processor.data['Date'], y=processor.data['RSI'], name='RSI', yaxis='y2'))
        fig_ind.update_layout(
            title="Price vs RSI",
            yaxis=dict(title="Price"),
            yaxis2=dict(title="RSI", overlaying='y', side='right')
        )
        st.plotly_chart(fig_ind, use_container_width=True)

        # 3. Prepare Data
        status_text.text("Preprocessing data for LSTM...")
        X_train, y_train, X_test, y_test, scaler = processor.prepare_data()
        progress_bar.progress(60)
        
        # 4. Build Model
        status_text.text("Building and training LSTM model...")
        input_shape = (X_train.shape[1], X_train.shape[2])
        model = build_lstm_model(input_shape)
        
        # Callback for progress (simple version)
        class CustomCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                status_text.text(f"Training Epoch {epoch+1}/{epochs} - Loss: {logs['loss']:.4f}")
        
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0, callbacks=[CustomCallback()])
        progress_bar.progress(80)
        
        # 5. Predict
        status_text.text("Generating predictions...")
        predictions = model.predict(X_test)
        
        # Inverse Transform
        # Create dummy array for inverse transform (since scaler expects 3 features)
        dummy_input = np.zeros((len(predictions), 3))
        dummy_input[:, 0] = predictions[:, 0]
        inverse_predictions = scaler.inverse_transform(dummy_input)[:, 0]
        
        dummy_y = np.zeros((len(y_test), 3))
        dummy_y[:, 0] = y_test
        inverse_actual = scaler.inverse_transform(dummy_y)[:, 0]
        
        progress_bar.progress(100)
        status_text.text("Done!")
        
        # 6. Visualization
        st.subheader("✅ Prediction Results")
        
        # Create a DataFrame for plotting
        # We need dates for the test set. 
        # The test set starts after train_size. 
        # sequence_length is 60. prepare_data splits by 0.8.
        # It's a bit tricky to get exact dates without returning them from processor, 
        # but we can approximation based on length.
        
        test_len = len(inverse_actual)
        test_dates = processor.data['Date'].iloc[-test_len:]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=test_dates, y=inverse_actual, mode='lines', name='Actual Price'))
        fig.add_trace(go.Scatter(x=test_dates, y=inverse_predictions, mode='lines', name='Predicted Price'))
        
        fig.update_layout(
            title=f"{ticker} Stock Price Prediction (Next Day Trend)",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            template="plotly_dark"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Metrics
        error = np.mean(np.abs(inverse_predictions - inverse_actual))
        st.metric("Mean Absolute Error (MAE)", f"${error:.2f}")

    except Exception as e:
        st.error(f"An error occurred: {e}")
