import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from prophet import Prophet
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split as surprise_train_test_split
from surprise import accuracy

# Load data
df = pd.read_csv("data/cleaned_data.csv")

# Aggregating data by customer and product
customer_product_data = df.groupby(['Client Name', 'Product']).agg({
    'Quantity': 'sum',
    'Amount': 'sum'
}).reset_index

df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

# Monthly purchase frequency and spending
df['YearMonth'] = df['Date'].dt.to_period('M')
monthly_purchase_data = df.groupby(['Client Name', 'Product', 'YearMonth']).agg({
    'Quantity': 'sum',
    'Amount': 'sum'
}).reset_index()

# Streamlit UI

st.title('Big Data Model Dashboard')


customer_name = st.selectbox("Select Customer", df['Client Name'].unique())
product_name = st.selectbox("Select Product", df[df['Client Name'] == customer_name]['Product'].unique())

customer_product_df = monthly_purchase_data[(monthly_purchase_data['Client Name'] == customer_name) & 
                                            (monthly_purchase_data['Product'] == product_name)]

prophet_df = customer_product_df[['YearMonth', 'Quantity']]
prophet_df.columns = ['ds', 'y']
prophet_df['ds'] = prophet_df['ds'].dt.to_timestamp()

# Section 1: Time Series Forecasting using Prophet
st.header("1. Time Series Forecasting (Product Purchase Forecast)")

model = Prophet()
model.fit(prophet_df)

# Make a future dataframe for predictions (next 3 months)
future = model.make_future_dataframe(periods=3, freq='M')
forecast = model.predict(future)

# Plot the forecast
fig1 = model.plot(forecast)
st.pyplot(fig1)

# Section 2: Customer Segmentation
st.header("2. Customer Segmentation (Total Amount vs Purchase Frequency)")

customer_features = df.groupby('Client Name').agg({
    'Quantity': 'sum',
    'Amount': 'sum',
    'Invoice No.': 'nunique'
}).reset_index()

customer_features.columns = ['Client Name', 'Total_Quantity', 'Total_Amount', 'Purchase_Frequency']

# Plot customer segmentation
fig2 = plt.figure(figsize=(10, 6))
sns.scatterplot(data=customer_features, x='Total_Amount', y='Purchase_Frequency', hue='Total_Quantity', 
                palette='viridis', size='Total_Quantity', sizes=(20, 200), legend=None)
plt.title("Customer Segmentation: Total Amount vs Purchase Frequency")
plt.xlabel("Total Amount Spent")
plt.ylabel("Purchase Frequency")
plt.grid(True)
st.pyplot(fig2)

# Section 3: Customer Lifetime Value (CLV)
st.header("3. Customer Lifetime Value (CLV)")

avg_margin = 0.3  # Example margin
retention_rate = 0.5  # Example retention rate
customer_features['CLV'] = (customer_features['Total_Amount'] * avg_margin) / (1 - retention_rate)

# Visualize CLV
fig3 = plt.figure(figsize=(10, 6))
sns.barplot(data=customer_features.sort_values('CLV', ascending=False).head(10), x='CLV', y='Client Name', palette='viridis')
plt.title("Top 10 Customers by Customer Lifetime Value (CLV)")
plt.xlabel("Customer Lifetime Value")
plt.ylabel("Client Name")
st.pyplot(fig3)

# Section 4: Upsell Prediction Model (Random Forest)
st.header("4. Upsell Prediction")

amount_threshold = customer_features['Total_Amount'].quantile(0.8)
customer_features['Upsell'] = (customer_features['Total_Amount'] > amount_threshold).astype(int)

features = customer_features[['Total_Quantity', 'Purchase_Frequency', 'Total_Amount']]
target = customer_features['Upsell']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.25, random_state=42)

# Train Random Forest Classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Calculate probability of upsell for each customer
customer_features['Upsell_Probability'] = rf.predict_proba(features)[:, 1]

# Show top 5 customers likely to upsell
top_upsell_customers = customer_features.sort_values('Upsell_Probability', ascending=False).head(5)
st.write("Top 5 Customers Likely to Buy an Upsell:")
st.write(top_upsell_customers[['Client Name', 'Upsell_Probability']])

# Section 5: Cross-Sell Prediction Model (Collaborative Filtering)
st.header("5. Cross-Sell Prediction (Collaborative Filtering)")

# Prepare data for Surprise model
reader = Reader(rating_scale=(0, df['Quantity'].max()))
data = Dataset.load_from_df(df[['Client Name', 'Product', 'Quantity']], reader)

trainset, testset = surprise_train_test_split(data, test_size=0.25, random_state=42)

algo = SVD()
algo.fit(trainset)

predictions = algo.test(testset)
rmse = accuracy.rmse(predictions)

st.write(f"Cross-Sell Model RMSE: {rmse}")

# Get recommendations for a specific customer
customer_inner_id = trainset.to_inner_uid(customer_name)
recommendations = []

for item_id in trainset.all_items():
    product_name = trainset.to_raw_iid(item_id)
    prediction = algo.predict(customer_inner_id, item_id)
    recommendations.append((product_name, prediction.est))

recommendations.sort(key=lambda x: x[1], reverse=True)

st.write(f"Top 5 Cross-Sell Recommendations for {customer_name}:")
for product, score in recommendations[:5]:
    st.write(f"{product}: Estimated Quantity = {score:.2f}")

# Final summary
st.header("Analysis Summary")
st.write("""
1. **Time Series Forecast:** Forecasted future purchase quantities using Prophet.
2. **Customer Segmentation:** Visualized customer engagement based on purchase frequency and total spending.
3. **Customer Lifetime Value (CLV):** Calculated CLV for each customer to identify valuable customers.
4. **Upsell Prediction:** Predicted likelihood of customers purchasing higher-value products using Random Forest.
5. **Cross-Sell Prediction:** Recommended products based on collaborative filtering.
""")
