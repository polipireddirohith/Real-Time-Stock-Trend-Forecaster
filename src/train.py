import argparse
import numpy as np
import tensorflow as tf
from data_processor import DataProcessor
from model import build_lstm_model
from utils import plot_predictions, plot_indicators

def train_and_predict(ticker, epochs=20, batch_size=32):
    # 1. Prepare Data
    processor = DataProcessor(ticker=ticker)
    processor.fetch_data()
    processor.add_technical_indicators()
    
    # Plot indicators before training
    # plot_indicators(processor.data, ticker)
    
    X_train, y_train, X_test, y_test, scaler = processor.prepare_data()
    
    # 2. Build Model
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_lstm_model(input_shape)
    
    # 3. Train
    print("Starting training...")
    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs)
    
    # 4. Predict
    predictions = model.predict(X_test)
    
    # Inverse transform predictions
    # We normalized with 3 features, so we need to inverse transform with 3 features.
    # Create a dummy array with same shape as original input to inverse transform
    
    # predictions is (n_samples, 1). We need (n_samples, 3) where column 0 is prediction.
    # The scaler expects 3 columns.
    
    # Construct array for inverse transformation
    # We take the X_test used for these predictions. X_test is (samples, seq_len, features)
    # We need to recover the other features (RSI, MACD) to inverse transform correctly, 
    # OR we can just create a dummy matrix since we only care about the Close price (col 0).
    
    # Helper to inverse transform only the target column
    # The scaler was fitted on [Close, RSI, MACD]
    # We have 'predictions' which are scaled 'Close' values.
    
    # Strategy:
    # 1. Create a matrix of zeros with shape (len(predictions), 3)
    # 2. Place predictions in the first column (index 0)
    # 3. Inverse transform
    # 4. Extract first column
    
    dummy_input = np.zeros((len(predictions), 3))
    dummy_input[:, 0] = predictions[:, 0]
    
    inverse_predictions = scaler.inverse_transform(dummy_input)
    final_predictions = inverse_predictions[:, 0]
    
    # Also inverse transform y_test to compare actual values
    dummy_y = np.zeros((len(y_test), 3))
    dummy_y[:, 0] = y_test
    inverse_actual = scaler.inverse_transform(dummy_y)
    final_actual = inverse_actual[:, 0]
    
    # 5. Visualize
    # Calculate train size for plotting (to know where test set starts)
    # prepare_data uses 0.8 split
    train_size = int(len(processor.data) * 0.8) # Approx
    
    # Actually, let's just pass the data object and plot it properly
    # We need to know the raw data corresponding to X_test
    
    # Simply plotting the test predictions vs actuals
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12,6))
    plt.plot(final_actual, label='Actual Data')
    plt.plot(final_predictions, label='Predicted Data')
    plt.title(f'{ticker} Stock Price Prediction')
    plt.legend()
    plt.show()

    # Save model
    model.save(f"{ticker}_lstm_model.h5")
    print(f"Model saved to {ticker}_lstm_model.h5")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LSTM for Stock Prediction")
    parser.add_argument("--ticker", type=str, default="AAPL", help="Stock Ticker Symbol")
    parser.add_argument("--epochs", type=int, default=5, help="Training Epochs")  # Small default for testing
    args = parser.parse_args()
    
    train_and_predict(args.ticker, args.epochs)
