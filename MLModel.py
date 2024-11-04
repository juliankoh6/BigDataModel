# Import necessary libraries
import pandas as pd
import numpy as np
from datetime import datetime
from prophet import Prophet
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split as surprise_train_test_split
from surprise import accuracy
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_excel("data/cleaned_data.xlsx")

# Aggregating data by customer and product
customer_product_data = df.groupby(['Client Name', 'Product']).agg({
    'Quantity': 'sum',
    'Amount': 'sum'
}).reset_index()

# Monthly purchase frequency and spending
df['YearMonth'] = df['Date'].dt.to_period('M')
monthly_purchase_data = df.groupby(['Client Name', 'Product', 'YearMonth']).agg({
    'Quantity': 'sum',
    'Amount': 'sum'
}).reset_index()

# ---------------------
# Section 1: Time Series Forecasting (Using Prophet)
customer_name = "101 Thai & Cafe (Wheeler Heights)"
product_name = "Bean Sprout"
customer_product_df = monthly_purchase_data[(monthly_purchase_data['Client Name'] == customer_name) &
                                            (monthly_purchase_data['Product'] == product_name)]

prophet_df = customer_product_df[['YearMonth', 'Quantity']]
prophet_df.columns = ['ds', 'y']
prophet_df['ds'] = prophet_df['ds'].dt.to_timestamp()

model = Prophet()
model.fit(prophet_df)

# Make a future dataframe for predictions (next 3 months)
future = model.make_future_dataframe(periods=3, freq='M')
forecast = model.predict(future)

# Plot the forecast
model.plot(forecast)
plt.title(f"3-Month Forecast for {product_name} for {customer_name}")
plt.show()

# Section 2: Customer Segmentation
# Calculate total quantity, total amount, and purchase frequency for each customer
customer_features = df.groupby('Client Name').agg({
    'Quantity': 'sum',
    'Amount': 'sum',
    'Invoice No.': 'nunique'
}).reset_index()
customer_features.columns = ['Client Name', 'Total_Quantity', 'Total_Amount', 'Purchase_Frequency']

# Customer segmentation based on Total Amount and Purchase Frequency
plt.figure(figsize=(10, 6))
sns.scatterplot(data=customer_features, x='Total_Amount', y='Purchase_Frequency', hue='Total_Quantity', palette='viridis', size='Total_Quantity', sizes=(20, 200), legend=None)
plt.title("Customer Segmentation: Total Amount vs Purchase Frequency")
plt.xlabel("Total Amount Spent")
plt.ylabel("Purchase Frequency")
plt.grid(True)
plt.show()

# ---------------------
# Section 3: Customer Lifetime Value (CLV) Calculation
# Assuming a fixed average margin and a retention rate
avg_margin = 0.3
retention_rate = 0.5
customer_features['CLV'] = (customer_features['Total_Amount'] * avg_margin) / (1 - retention_rate)

# Visualize CLV
plt.figure(figsize=(10, 6))
sns.barplot(data=customer_features.sort_values('CLV', ascending=False).head(10), x='CLV', y='Client Name', palette='viridis')
plt.title("Top 10 Customers by Customer Lifetime Value (CLV)")
plt.xlabel("Customer Lifetime Value")
plt.ylabel("Client Name")
plt.show()

# ---------------------
# Section 4: Upsell Prediction Model (Random Forest Classifier)
# Create a synthetic binary target variable (for demonstration purposes)
np.random.seed(0)
customer_features['Upsell_Response'] = np.random.choice([0, 1], size=len(customer_features))

X = customer_features[['Total_Quantity', 'Total_Amount', 'Purchase_Frequency']]
y = customer_features['Upsell_Response']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print("Upsell Prediction Model Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report for Upsell Prediction:\n", classification_report(y_test, y_pred))

# ---------------------
# Section 5: Cross-Sell Prediction Model (Collaborative Filtering)
reader = Reader(rating_scale=(0, df['Quantity'].max()))
data = Dataset.load_from_df(df[['Client Name', 'Product', 'Quantity']], reader)

trainset, testset = surprise_train_test_split(data, test_size=0.25, random_state=42)

algo = SVD()
algo.fit(trainset)

predictions = algo.test(testset)
print("Cross-Sell Model RMSE:", accuracy.rmse(predictions))

# Cross-sell recommendations for a specific customer
customer_inner_id = trainset.to_inner_uid(customer_name)
recommendations = []

for item_id in trainset.all_items():
    product_name = trainset.to_raw_iid(item_id)
    prediction = algo.predict(customer_inner_id, item_id)
    recommendations.append((product_name, prediction.est))

recommendations.sort(key=lambda x: x[1], reverse=True)
print(f"Top 5 Cross-Sell Recommendations for {customer_name}:")
for product, score in recommendations[:5]:
    print(f"{product}: Estimated Quantity = {score:.2f}")

# ---------------------
# Analysis Summary
# ---------------------
print("\nSummary of Analyses:")
print("1. Time Series Forecast: Forecasted future purchase quantities using Prophet.")
print("2. Customer Segmentation: Visualized customer engagement based on purchase frequency and total spending.")
print("3. Customer Lifetime Value (CLV): Calculated CLV for each customer to identify valuable customers.")
print("4. Upsell Prediction: Predicted customers likely to respond to upsell using Random Forest.")
print("5. Cross-Sell Prediction: Recommended products for cross-selling using Collaborative Filtering (SVD).")