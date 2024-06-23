import streamlit as st
import pickle
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor


with open('random_forest_model.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)


with open('minmax_scaler.pkl', 'rb') as scaler_file:
    loaded_scaler = pickle.load(scaler_file)

# Define the features
features = ["value_eur", "age", "potential", "wage_eur", "movement_reactions", "defending", "mentality_composure", "skill_ball_control"]

# Function to get user input
def user_input_features():
    user_data = {}
    for feature in features:
        user_data[feature] = st.sidebar.number_input(f"Enter {feature}", value=0)
    return pd.DataFrame(user_data, index=[0])

# Get user input
input_df = user_input_features()

# Scale the input data
input_scaled = loaded_scaler.transform(input_df)


# Display input data
st.write('## Input Data')
st.write(input_df)


# Predict using the loaded model
if st.button('Predict'):
    prediction = loaded_model.predict(input_scaled)
    st.write(f"Predicted Player Rating: {prediction[0]}")