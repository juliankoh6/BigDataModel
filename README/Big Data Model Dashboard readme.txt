Big Data Model Dashboard

Features
Time Series Forecasting:
Predicts future product purchase quantities using Facebook's Prophet model.
Helps identify trends for demand forecasting.

Customer Segmentation:
Visualizes customer purchase patterns based on total spending and purchase frequency.
Segments customers to understand their behavior better.

Customer Lifetime Value (CLV):
Calculates and ranks customers based on their projected lifetime value.

Upsell Prediction:
Identifies customers likely to purchase higher-value products using a Random Forest Classifier.
Cross-Sell Prediction:

Recommends products to customers using a collaborative filtering approach (SVD).
Setup Instructions

1. Ensure the following libraries are installed:
pandas
numpy
matplotlib
seaborn
streamlit
prophet
scikit-learn
surprise

use command:
pip install pandas numpy matplotlib seaborn streamlit prophet scikit-learn scikit-surprise

2. Data Requirements
Input data file: data/cleaned_data.csv

The data should include the following columns:
Client Name: Name of the customer.
Product: Name of the product purchased.
Quantity: Number of units purchased.
Amount: Total spending on the product.
Date: Date of purchase (format: YYYY-MM-DD).
Invoice No.: Unique identifier for the transaction.

3. Running the Dashboard
Run the Streamlit dashboard using the following command:
streamlit run dashboard.py


Customization
filepath 
avg_margin and retention_rate variables in the CLV section.
amount_threshold in the upsell prediction section to change the threshold for high-value customers.
