import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from train import train_and_predict

if __name__ == "__main__":
    ticker = input("Enter Stock Ticker (e.g., AAPL): ") or "AAPL"
    try:
        train_and_predict(ticker, epochs=10)
    except Exception as e:
        print(f"An error occurred: {e}")
