import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_predictions(data, train_size, y_test, predictions, title="Stock Price Prediction"):
    """
    Plots the valid data vs predictions.
    """
    # Adjust valid data to match the dates
    valid = data.iloc[train_size+60:].copy() # Adjust for sequence_length
    
    # Check if lengths match, if not, slice to match
    min_len = min(len(valid), len(predictions))
    valid = valid.iloc[:min_len]
    predictions = predictions[:min_len]

    valid['Predictions'] = predictions

    plt.figure(figsize=(16,8))
    plt.title(title)
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price USD ($)', fontsize=18)
    
    # Plot training data mostly for context, but maybe just test data is clearer
    # valid_all = data.iloc[:].copy()
    # plt.plot(valid_all['Close'])
    
    plt.plot(valid['Close'], label='Actual Price')
    plt.plot(valid['Predictions'], label='Predicted Price')
    plt.legend(loc='lower right')
    plt.show()

def plot_indicators(data, ticker):
    plt.figure(figsize=(14, 8))
    
    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(data['Close'], label='Close Price')
    ax1.set_title(f'{ticker} Price')
    ax1.legend()
    
    ax2 = plt.subplot(2, 1, 2)
    ax2.plot(data['RSI'], label='RSI', color='orange')
    ax2.axhline(70, linestyle='--', color='red')
    ax2.axhline(30, linestyle='--', color='green')
    ax2.set_title('Relative Strength Index (RSI)')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()
