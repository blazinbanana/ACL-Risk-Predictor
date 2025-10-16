from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization

def create_acl_risk_model(sequence_length, num_features, num_classes=2):
    """
    Create LSTM model for ACL risk classification
    """
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(sequence_length, num_features)),
        Dropout(0.3),
        BatchNormalization(),
        
        LSTM(32, return_sequences=True),
        Dropout(0.3),
        BatchNormalization(),
        
        LSTM(16),
        Dropout(0.3),
        
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model