import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class DataProcessor:
    def __init__(self, ticker, start_date=None, end_date=None, period="2y"):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.period = period
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.data = None

    def fetch_data(self):
        """Fetches historical data from Yahoo Finance."""
        print(f"Fetching data for {self.ticker}...")
        if self.start_date and self.end_date:
            self.data = yf.download(self.ticker, start=self.start_date, end=self.end_date, progress=False)
        else:
            self.data = yf.download(self.ticker, period=self.period, progress=False)
        
        if self.data.empty:
            raise ValueError("No data fetched. Check ticker symbol or internet connection.")
        
        # Ensure single level column index if MultiIndex
        if isinstance(self.data.columns, pd.MultiIndex):
            self.data.columns = self.data.columns.get_level_values(0)

        self.data.reset_index(inplace=True)
        print(f"Data fetched: {len(self.data)} rows.")

    def add_technical_indicators(self):
        """Calculates and adds RSI and MACD to the dataframe."""
        if self.data is None:
            raise ValueError("Data not fetched yet. Call fetch_data() first.")

        df = self.data.copy()
        
        # Calculate RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).ewm(alpha=1/14, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # Calculate MACD
        exp12 = df['Close'].ewm(span=12, adjust=False).mean()
        exp26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp12 - exp26
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

        df.dropna(inplace=True)
        self.data = df
        print("Technical indicators added.")

    def prepare_data(self, sequence_length=60, train_split=0.8):
        """
        Prepares data for LSTM model.
        Returns:
            X_train, y_train, X_test, y_test, scaler
        """
        if self.data is None:
            raise ValueError("Data not processed yet.")

        # Features to use: Close price, RSI, MACD
        features = ['Close', 'RSI', 'MACD']
        dataset = self.data[features].values

        # Scale the data
        scaled_data = self.scaler.fit_transform(dataset)

        # Create sequences
        X = []
        y = []
        
        # We want to predict the 'Close' price (index 0)
        target_col_index = 0

        for i in range(sequence_length, len(scaled_data)):
            X.append(scaled_data[i-sequence_length:i])
            y.append(scaled_data[i, target_col_index])

        X, y = np.array(X), np.array(y)

        # Split into train and test
        train_size = int(len(X) * train_split)
        
        X_train = X[:train_size]
        y_train = y[:train_size]
        X_test = X[train_size:]
        y_test = y[train_size:]

        print(f"Training Data Shape: {X_train.shape}, {y_train.shape}")
        print(f"Testing Data Shape: {X_test.shape}, {y_test.shape}")
        
        return X_train, y_train, X_test, y_test, self.scaler
