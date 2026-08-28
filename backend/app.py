# Import necessary libraries
import numpy as np
import joblib  # For loading the serialized model
import pandas as pd  # For data manipulation
from flask import Flask, request, jsonify  # For creating the Flask API

# Initialize the Flask application
superkart_api = Flask("SuperKart Sales Predictor") # Changed app name from Airbnb to SuperKart

# Load the trained machine learning model
model = joblib.load("superkart_model.joblib") # Changed model filename

# Define a route for the home page (GET request)
@superkart_api.get('/')
def home():
    """
    This function handles GET requests to the root URL ('/') of the API.
    It returns a simple welcome message.
    """
    return "Welcome to the SuperKart Sales Prediction API!"

# Define an endpoint for single property prediction (POST request)
@superkart_api.post('/v1/predict') # Changed endpoint from /v1/rental to /v1/predict
def predict_sales_price(): # Changed function name from predict_rental_price to predict_sales_price
    """
    This function handles POST requests to the '/v1/predict' endpoint.
    It expects a JSON payload containing product and store details and returns
    the predicted sales price as a JSON response.
    """
    # Get the JSON data from the request body
    product_data = request.get_json()

    # Extract relevant features from the JSON data
    sample = {
        'Product_Weight': product_data['Product_Weight'],
        'Product_Sugar_Content': product_data['Product_Sugar_Content'],
        'Product_Allocated_Area': product_data['Product_Allocated_Area'],
        'Product_MRP': product_data['Product_MRP'],
        'Product_Type': product_data['Product_Type'],
        'Store_Establishment_Year': product_data['Store_Establishment_Year'],
        'Store_Location_City_Type': product_data['Store_Location_City_Type'],
        'Store_Type': product_data['Store_Type'],
        'Store_Size': product_data['Store_Size']
    }

    # Convert the extracted data into a Pandas DataFrame
    input_data = pd.DataFrame([sample])

    # Make prediction
    predicted_sales = model.predict(input_data)[0]

    # Convert predicted_sales to Python float
    predicted_sales = round(float(predicted_sales), 2)

    # Return the predicted sales
    return jsonify({'Predicted Sales Total': predicted_sales})

# Define an endpoint for batch prediction (POST request)
@superkart_api.post('/v1/predictbatch') # Changed endpoint from /v1/rentalbatch to /v1/predictbatch
def predict_sales_price_batch(): # Changed function name from predict_rental_price_batch to predict_sales_price_batch
    """
    This function handles POST requests to the '/v1/predictbatch' endpoint.
    It expects a CSV file containing product and store details for multiple entries
    and returns the predicted sales prices as a dictionary in the JSON response.
    """
    # Get the uploaded CSV file from the request
    file = request.files['file']

    # Read the CSV file into a Pandas DataFrame
    input_data = pd.read_csv(file)

    # Make predictions for all entries in the DataFrame
    predicted_sales_list = model.predict(input_data).tolist()

    # Create a dictionary of predictions with a generic ID as keys
    # Assuming the input CSV might not have a specific 'id' column for products
    # Using index as a simple identifier for batch predictions
    output_dict = {f'Product_{i}': round(float(sales), 2) for i, sales in enumerate(predicted_sales_list)}

    # Return the predictions dictionary as a JSON response
    return jsonify(output_dict)

# Run the Flask application in debug mode if this script is executed directly
if __name__ == '__main__':
    superkart_api.run(debug=True)
