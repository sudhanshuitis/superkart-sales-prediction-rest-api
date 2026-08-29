# Import necessary libraries
import sys
import numpy as np
import joblib  # For loading the serialized model
import pandas as pd  # For data manipulation
from flask import Flask, request, jsonify  # For creating the Flask API
from sklearn.base import BaseEstimator, TransformerMixin

# 1. Add custom transformer definition so joblib can unpickle the model pipeline
class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, current_year=2026):
        self.current_year = current_year

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X['Store_Age'] = self.current_year - X['Store_Establishment_Year']
        X['Product_Category'] = X['Product_Id'].str[:2]
        X['Product_Sugar_Content'] = X['Product_Sugar_Content'].replace({'reg': 'Regular'})
        perishable = [
            'Dairy', 'Meat', 'Fruits and Vegetables', 'Breakfast',
            'Seafood', 'Frozen Foods', 'Breads'
        ]
        X['Type_of_Food'] = X['Product_Type'].apply(
            lambda pt: 'Perishable' if pt in perishable else 'Non-Perishable'
        )
        X = X.drop(columns=['Product_Id', 'Store_Establishment_Year'])
        return X

# 2. Inject class into __main__ so Gunicorn workers locate it
sys.modules['__main__'].FeatureEngineer = FeatureEngineer

# Initialize the Flask application
superkart_sales_forecast_api = Flask("SuperKart Sales Data Forecasting")

# 3. Add alias for Gunicorn (looks for 'app' by default)
app = superkart_sales_forecast_api

# Load the trained machine learning model (handles local and container paths)
model_name = "superkart_sales_forecasting_model_v1_0.joblib"
try:
    model = joblib.load(model_name)
except FileNotFoundError:
    model = joblib.load(f"backend_files/{model_name}")

# Define a route for the home page (GET request)
@superkart_sales_forecast_api.get('/')
def home():
    """
    This function handles GET requests to the root URL ('/') of the API.
    It returns a simple welcome message.
    """
    return "Welcome to the SuperKart Sales Data Forecasting API!"

# Define an endpoint for single product forecasting (POST request)
@superkart_sales_forecast_api.post('/v1/sales')
def forecast_single_sales_data():
    """
    This function handles POST requests to the '/v1/sales' endpoint.
    It expects a JSON payload containing product and related details (original features)
    and returns the projected sales estimate as a JSON response.
    """
    # Get the JSON data from the request body
    product_data = request.get_json()

    # Extract relevant features from the JSON data
    sample = {
        'Product_Id': product_data['Product_Id'],
        'Product_Weight': product_data['Product_Weight'],
        'Product_Sugar_Content': product_data['Product_Sugar_Content'],
        'Product_Allocated_Area': product_data['Product_Allocated_Area'],
        'Product_Type': product_data['Product_Type'],
        'Product_MRP': product_data['Product_MRP'],
        'Store_Id': product_data['Store_Id'],
        'Store_Establishment_Year': product_data['Store_Establishment_Year'],
        'Store_Size': product_data['Store_Size'],
        'Store_Location_City_Type': product_data['Store_Location_City_Type'],
        'Store_Type': product_data['Store_Type']
    }

    # Convert the extracted data into a Pandas DataFrame
    input_data = pd.DataFrame([sample])

    # Make prediction
    predicted_sales = model.predict(input_data)[0]

    # Convert predicted_sales to Python float and round
    predicted_sales = round(float(predicted_sales), 2)

    # 4. Return the key expected by the Streamlit frontend
    return jsonify({'Predicted Sales (in dollars)': predicted_sales})


# Define an endpoint for batch prediction (POST request)
@superkart_sales_forecast_api.post('/v1/salesbatch')
def predict_batch_sales():
    """
    This function handles POST requests to the '/v1/salesbatch' endpoint.
    It expects a CSV file containing product details for multiple products (original features)
    and returns the predicted sales as a list in the JSON response.
    """
    # Get the uploaded CSV file from the request
    file = request.files['file']

    # Read the CSV file into a Pandas DataFrame. The pipeline expects original feature names.
    input_data = pd.read_csv(file)

    # Make predictions for all products in the DataFrame
    predicted_sales_array = model.predict(input_data)

    # Convert predictions to a list of Python floats and round
    predicted_sales_list = [round(float(s), 2) for s in predicted_sales_array]

    # Create a dictionary of predictions with Product IDs as keys
    Product_Ids = input_data['Product_Id'].tolist()  # Assuming 'Product_Id' is the product ID column
    output_dict = dict(zip(Product_Ids, predicted_sales_list))  # Use actual sales

    # Return the predictions dictionary as a JSON response
    return output_dict

# Run the Flask application in debug mode if this script is executed directly
if __name__ == '__main__':
    superkart_sales_forecast_api.run(debug=True)
