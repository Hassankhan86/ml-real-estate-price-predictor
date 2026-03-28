import streamlit as st
import pickle
import pandas as pd
import numpy as np
import os

st.set_page_config(page_title="Property Price Demo", layout="wide")

# Load data and pipeline
with open('artifacts/df.pkl','rb') as file:
    df = pickle.load(file)

with open('artifacts/pipeline.pkl','rb') as file:
    pipeline = pickle.load(file)

# print(df)
# st.header('Enter your inputs')

st.title("🏡 Real Estate Price Predictor")
st.markdown("Please fill in the details of the property below:")

# Add spacing
st.write("---")


# Using columns to make layout compact
col1, col2, col3 = st.columns(3)

with col1:
    property_type = st.selectbox('Property Type', ['flat', 'house'])
    bedrooms = float(st.selectbox('Number of Bedrooms', sorted(df['bedRoom'].unique().tolist())))
    bathroom = float(st.selectbox('Number of Bathrooms', sorted(df['bathroom'].unique().tolist())))
    # bedrooms = st.slider('Number of Bedrooms', min_value=int(df['bedRoom'].min()), max_value=int(df['bedRoom'].max()), step=1)
    # bathroom = st.slider('Number of Bathrooms', min_value=int(df['bathroom'].min()), max_value=int(df['bathroom'].max()), step=1)
    balcony = st.selectbox('Balconies', sorted(df['balcony'].unique().tolist()))

with col2:
    sector = st.selectbox('Sector', sorted(df['sector'].unique().tolist()))
    property_age = st.selectbox('Property Age', sorted(df['agePossession'].unique().tolist()))
    built_up_area = st.number_input('Built Up Area (sqft)', min_value=0.0, step=50.0)
    # built_up_area = float(st.number_input('Built Up Area'))
    floor_category = st.selectbox('Floor Category', sorted(df['floor_category'].unique().tolist()))

with col3:
    servant_room = float(st.selectbox('Servant Room', [0.0, 1.0]))
    store_room = float(st.selectbox('Store Room', [0.0, 1.0]))
    furnishing_type = st.selectbox('Furnishing Type', sorted(df['furnishing_type'].unique().tolist()))
    luxury_category = st.selectbox('Luxury Category', sorted(df['luxury_category'].unique().tolist()))


st.markdown("---")

if st.button('Predict'):

    # form a dataframe
    data = [[property_type, sector, bedrooms, bathroom, balcony, property_age, built_up_area, servant_room, store_room, furnishing_type, luxury_category, floor_category]]
    columns = ['property_type', 'sector', 'bedRoom', 'bathroom', 'balcony',
               'agePossession', 'built_up_area', 'servant room', 'store room',
               'furnishing_type', 'luxury_category', 'floor_category']

    # Convert to DataFrame
    one_df = pd.DataFrame(data, columns=columns)

    #st.dataframe(one_df)

    # predict
    base_price = np.expm1(pipeline.predict(one_df))[0]
    low = float(base_price) - 0.22
    high = float(base_price) + 0.22
    
    # base_price = float(base_price)
    st.text(f"Price: {float(base_price):.2f}")
    st.text("Price: {}".format(round(base_price, 2)))
    
    st.text("The price of the flat is between {} Cr and {} Cr".format(round(low,2),round(high,2)))