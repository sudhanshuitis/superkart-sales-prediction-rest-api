import streamlit as st
import pandas as pd
import requests

# Base URL of the Flask backend
BACKEND_URL = "http://backend:7860"

# Set the title of the Streamlit app
st.title("SuperKart Sales Data Forecasting App")

# Section for online prediction
st.subheader("Online Prediction")


# Collect user input for SuperKart product and store features
product_type = st.selectbox(
    "Product Type",
    [
        "Fruits and Vegetables", "Snack Foods", "Frozen Foods", "Dairy",
        "Household", "Baking Goods", "Canned", "Health and Hygiene",
        "Meat", "Soft Drinks", "Breads", "Hard Drinks",
        "Others", "Starchy Foods", "Breakfast", "Seafood"
    ]
)
product_sugar_content = st.selectbox(
    "Product Sugar Content",
    ["Low Sugar", "Regular", "No Sugar"]
)
product_weight = st.number_input(
    "Product Weight",
    min_value=4.0,
    max_value=22.0,
    step=0.1,
    value=12.5
)
product_allocated_area = st.number_input(
    "Product Allocated Area",
    min_value=0.000,
    max_value=0.300,
    step=0.001,
    format="%.3f",
    value=0.065
)
product_mrp = st.number_input(
    "Product MRP",
    min_value=30.0,
    max_value=270.0,
    step=1.0,
    value=147.0
)
store_id = st.selectbox(
    "Store Id",
    ["OUT004", "OUT001", "OUT003", "OUT002"]
)
store_establishment_year = st.number_input(
    "Store Establishment Year",
    min_value=1985,
    max_value=2015,
    step=1,
    value=2002
)
store_size = st.selectbox(
    "Store Size",
    ["Small", "Medium", "High"]
)
store_location_city_type = st.selectbox(
    "Store Location City Type",
    ["Tier 1", "Tier 2", "Tier 3"]
)
store_type = st.selectbox(
    "Store Type",
    ["Supermarket Type2", "Supermarket Type1", "Departmental Store", "Food Mart"]
)

# 1. Derive matching Product_Id prefix so the backend pipeline can extract Product_Category
drink_types = ['Soft Drinks', 'Hard Drinks']
non_consumables = ['Household', 'Health and Hygiene', 'Others']

if product_type in drink_types:
    product_id = 'DR001'
elif product_type in non_consumables:
    product_id = 'NC001'
else:
    product_id = 'FD001'

# Convert user input into a DataFrame
# 2. Added 'Product_Id' and fixed casing to 'Product_MRP'
input_data = pd.DataFrame([{
    'Product_Id': product_id,
    'Product_Type': product_type,
    'Product_Sugar_Content': product_sugar_content,
    'Product_Weight': product_weight,
    'Product_Allocated_Area': product_allocated_area,
    'Product_MRP': product_mrp,
    'Store_Id': store_id,
    'Store_Establishment_Year': store_establishment_year,
    'Store_Size': store_size,
    'Store_Location_City_Type': store_location_city_type,
    'Store_Type': store_type
}])


# Make prediction when the "Predict" button is clicked
if st.button("Predict", type="primary"):
    response = requests.post(f"{BACKEND_URL}/v1/sales", json=input_data.to_dict(orient='records')[0])  # Send data to Flask API
    if response.status_code == 200:
        prediction = response.json()['Predicted Sales (in dollars)']
        st.success(f"Predicted Product Sales (in dollars): {prediction}")
    else:
        st.error("Unable to connect to the prediction API.")

# Section for batch prediction
st.subheader("Batch Prediction")

# Allow users to upload a CSV file for batch prediction
uploaded_file = st.file_uploader("Upload CSV file for batch prediction", type=["csv"])

# Make batch prediction when the "Predict Batch" button is clicked
if uploaded_file is not None:
    if st.button("Predict Batch", type="primary"):
        response = requests.post(f"{BACKEND_URL}/v1/salesbatch", files={"file": uploaded_file})  # Send file to Flask API
        if response.status_code == 200:
            predictions = response.json()
            st.success("Batch predictions completed!")
            st.write(predictions)  # Display the predictions
        else:
            st.error("Unable to connect to the prediction API.")
