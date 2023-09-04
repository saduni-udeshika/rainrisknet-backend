import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, SimpleRNNCell, LSTMCell
from keras.optimizers import Adam
from keras.utils import to_categorical
import random
import pickle  # Import the pickle library for saving label encoders

# Load the dataset
data = pd.read_csv('forecastmodeldata.csv')  

# Data Preprocessing
features = ['Rainfalls_mm', 'Relative_Humidity_Day', 'Relative_Humidity_Night',
            'Temperature_Min', 'Temperature_Max', 'Average_Wind_Speed_km/h']
target_cols = ['Disaster Occurrence', 'Severity']

# Convert date to datetime format
data['Date'] = pd.to_datetime(data['Date'])

# Encode categorical features
label_encoders = {}
for col in ['Land Cover', 'Landform', 'Disaster Type']:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Train-Test Split
train_size = int(0.8 * len(data))
train_data = data.iloc[:train_size]
test_data = data.iloc[train_size:]

# Normalize features
scaler = StandardScaler()
train_data.loc[:, features] = scaler.fit_transform(train_data[features])
test_data.loc[:, features] = scaler.transform(test_data[features])

# Model Training
X_train = train_data[features].values
y_train_occurrence = to_categorical(train_data['Disaster Occurrence'].apply(lambda x: 1 if x == 'Yes' else 0))

# Reshape input data for RNN
X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))  # (batch_size, time_steps=1, features)

model = Sequential()
model.add(LSTM(128, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(128, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(128))
model.add(Dense(2, activation='softmax'))  # Adjust the number of units for output classes

model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train_occurrence, epochs=50, batch_size=32, validation_split=0.1)

# Save the trained model
model.save('disaster_forecast_model.h5')

# Save the label encoders using pickle
with open('disaster_forecast_label_encoders.pkl', 'wb') as le_file:
    pickle.dump(label_encoders, le_file)
