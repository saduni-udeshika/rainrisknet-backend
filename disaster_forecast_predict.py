import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from keras.models import load_model
from geopy.distance import geodesic
import random
import sys
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS, cross_origin
import pickle 

# Load the label encoders
with open('disaster_forecast_label_encoders.pkl', 'rb') as le_file:
    label_encoders = pickle.load(le_file)

# Load the trained model
model = load_model('disaster_forecast_model.h5')

# Load the dataset
data = pd.read_csv('forecastmodeldata.csv')  # Replace with your actual file path

# Data Preprocessing
features = ['Rainfalls_mm', 'Relative_Humidity_Day', 'Relative_Humidity_Night',
            'Temperature_Min', 'Temperature_Max', 'Average_Wind_Speed_km/h']

# Normalize features
scaler = StandardScaler()
data.loc[:, features] = scaler.fit_transform(data[features])

# Data Preprocessing
features = ['Rainfalls_mm', 'Relative_Humidity_Day', 'Relative_Humidity_Night',
            'Temperature_Min', 'Temperature_Max', 'Average_Wind_Speed_km/h']

# Define your create_input_data function here
def create_input_data(date, location, randomize=True):
    if randomize:
        random.seed(hash((date, location)))
        rainfalls = random.uniform(0.0, 50.2)
        humidity_day = random.uniform(50.1, 100.4)
        humidity_night = random.uniform(50.2, 100.4)  # Add this line for 'Relative_Humidity_Night'
        temperature_min = random.uniform(20.0, 35.5)  # Add this line for 'Temperature_Min'
        temperature_max = random.uniform(20.0, 35.2)
        wind_speed = random.uniform(0.0, 19.2)  # Add this line for 'Average_Wind_Speed_km/h'
        latitude = random.uniform(6.0, 7.1)
        longitude = random.uniform(80.0, 81.1)
    else:
        # Use provided values for non-randomized input
        rainfalls = 10.5
        humidity_day = 75.2
        humidity_night = 70.4  # Add this line for 'Relative_Humidity_Night'
        temperature_min = 28.6  # Add this line for 'Temperature_Min'
        temperature_max = 32.5
        wind_speed = 5.2  # Add this line for 'Average_Wind_Speed_km/h'
        latitude = 6.6828
        longitude = 80.3992

    input_data = {
        'Date': [date],
        'Location': [location],
        'Rainfalls_mm': [rainfalls],
        'Relative_Humidity_Day': [humidity_day],
        'Relative_Humidity_Night': [humidity_night],  # Add this line
        'Temperature_Min': [temperature_min],  # Add this line
        'Temperature_Max': [temperature_max],
        'Average_Wind_Speed_km/h': [wind_speed],  # Add this line
        'Latitude': [latitude],
        'Longitude': [longitude]
        # ... add other features
    }

    input_df = pd.DataFrame(input_data)
    return input_df


# Nearest Shelter Calculation
def calculate_nearest_shelter(lat, lon, shelter_locations):
    nearest_shelter = min(shelter_locations, key=lambda shelter: geodesic((lat, lon), shelter).kilometers)
    return nearest_shelter[0]  # Return the first element of the nearest_shelter tuple (latitude)


def predict_disaster(date, location):
    input_data = create_input_data(date, location)

    # Ensure this features list matches the features in the create_input_data function
    features = ['Rainfalls_mm', 'Relative_Humidity_Day', 'Relative_Humidity_Night',
                'Temperature_Min', 'Temperature_Max', 'Average_Wind_Speed_km/h']

    input_features = input_data[features].values

    # Flatten the input features for standardization
    input_features_flattened = input_features.reshape((input_features.shape[0], -1))

    # Standardize the flattened input features using the same scaler as used during training
    scaler = StandardScaler()
    input_features_flattened = scaler.fit_transform(input_features_flattened)

    # Reshape the standardized flattened features back to the desired shape
    input_features = input_features_flattened.reshape((1, input_features.shape[0], input_features.shape[1]))

    # Make predictions using the model
    predictions = model.predict(input_features)

    # Define shelter locations (latitude, longitude) in your area
    shelter_locations = data[['Latitude', 'Longitude']].values  # Use the 'Latitude' and 'Longitude' columns from your dataset

    # Get latitude and longitude from input_data
    lat = input_data['Latitude'].values[0]
    lon = input_data['Longitude'].values[0]

    # Calculate nearest shelter
    nearest_shelter = calculate_nearest_shelter(lat, lon, shelter_locations)
    nearest_shelter_km = round(nearest_shelter, 2)  # Round to two decimal places

    # Format the nearest_shelter_km value to have exactly 2 decimal places
    nearest_shelter_formatted = "{:.2f}".format(nearest_shelter_km)

    # Make predictions using the model
    predictions = model.predict(input_features)

    # Get the prediction for 'Disaster Occurrence'
    predicted_disaster_occurrence = predictions[0][0]

    # Define random threshold for disaster occurrence
    random_threshold_occurrence = random.uniform(0.0, 1.0)

    if predicted_disaster_occurrence > random_threshold_occurrence:
        disaster_occurrence = 'Yes'
        # Introduce randomness to the prediction process for 'Disaster Type'
        random_disaster_type = random.uniform(0.0, 1.0)
        if random_disaster_type < 0.5:
            disaster_type_idx = 0  # Flood
        else:
            disaster_type_idx = 1  # Landslide

        disaster_type_encoder = label_encoders['Disaster Type']
        if hasattr(disaster_type_encoder, 'classes_') and len(disaster_type_encoder.classes_):
            disaster_type = disaster_type_encoder.inverse_transform([disaster_type_idx])[0]
            if disaster_type == 0:
                disaster_type = 'Flood'
            elif disaster_type == 1:
                disaster_type = 'Landslide'
        else:
            disaster_type = f'Disaster Type {disaster_type_idx}'

        # Introduce randomness to the prediction process for 'Severity'
        severity_probs = predictions[0][3:6]
        if len(severity_probs) > 0:
            normalized_severity_probs = np.exp(severity_probs) / np.sum(np.exp(severity_probs))
            random_severity_idx = np.random.choice(len(normalized_severity_probs), p=normalized_severity_probs)
            severity_labels = ['No Disaster', 'Low', 'Medium', 'High']
            severity = severity_labels[random_severity_idx]
        else:
            severity = 'Low'  # Default value when severity_probs is empty

    else:
        disaster_occurrence = 'No'
        disaster_type = 'No Disaster'
        severity = 'No Disaster'

    # Format the output values to have exactly 2 decimal places
    wind_speed = "{:.2f}".format(input_data['Average_Wind_Speed_km/h'].values[0])
    temperature_max = "{:.2f}".format(input_data['Temperature_Max'].values[0])
    rainfalls = "{:.2f}".format(input_data['Rainfalls_mm'].values[0])
    humidity_day = "{:.2f}".format(input_data['Relative_Humidity_Day'].values[0])
    nearest_shelter_km = "{:.2f}".format(nearest_shelter_km)  # Format the nearest shelter distance

    output = {
        'Disaster Occurrence': disaster_occurrence,
        'Disaster Type': disaster_type,
        'Severity': severity,
        'Disaster Date': date,
        'Location': location,
        'Wind Speed (km/h)': wind_speed,
        'Temperature Max (°C)': temperature_max,
        'Rainfalls (mm)': rainfalls,
        'Humidity Day (%)': humidity_day,
    }
    # Conditionally add 'Nearest Shelter_km' field for disaster occurrence
    if disaster_occurrence == 'Yes':
        output['Nearest Shelter_km'] = "{:.2f}".format(float(nearest_shelter))

    return output

if __name__ == "__main__":
    input_date = input("Enter the date (YYYY-MM-DD): ")
    input_location = input("Enter the location: ")

    if input_location in data['Location'].values:
        input_lat = data[data['Location'] == input_location]['Latitude'].values[0]
        input_lon = data[data['Location'] == input_location]['Longitude'].values[0]
    else:
        print(f"Location '{input_location}' not available.")
        sys.exit(1)

    # Load or create the label encoders
    output = predict_disaster(input_date, input_location)
    
    print("Disaster Prediction Results:")
    print("Date:", output['Disaster Date'])
    print("Location:", output['Location'])
    print("Latitude:", input_lat)
    print("Longitude:", input_lon)
    print("Wind Speed (km/h):", output['Wind Speed (km/h)'])
    print("Temperature Max (°C):", output['Temperature Max (°C)'])
    print("Rainfalls (mm):", output['Rainfalls (mm)'])
    print("Humidity Day (%):", output['Humidity Day (%)'])
    print("Disaster Occurrence:", output['Disaster Occurrence'])
    if output['Disaster Occurrence'] == 'Yes':
        print("Disaster Type:", output['Disaster Type'])
        print("Severity:", output['Severity'])
        print("Nearest Shelter Distance (km):", output['Nearest Shelter_km'])