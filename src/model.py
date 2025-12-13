from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

def build_lstm_model(input_shape):
    """
    Builds and compiles the LSTM model.
    """
    model = Sequential()
    
    # First LSTM layer with Dropout
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    
    # Second LSTM layer with Dropout
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    
    # Dense layers
    model.add(Dense(units=25))
    model.add(Dense(units=1))
    
    # Compile model
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    return model
