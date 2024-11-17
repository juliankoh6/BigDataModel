import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from prophet import Prophet
from mlxtend.frequent_patterns import fpgrowth, association_rules
import streamlit as st

@st.cache
def load_data():
    df = pd.read_csv("data/cleaned_data.csv")
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['YearMonth'] = df['Date'].dt.to_period('M')
    return df

df = load_data()

customer_product_data = df.groupby(['Client Name', 'Product']).agg({
    'Quantity': 'sum',
    'Amount': 'sum'
}).reset_index()

monthly_purchase_data = df.groupby(['Client Name', 'Product', 'YearMonth']).agg({
    'Quantity': 'sum',
    'Amount': 'sum'
}).reset_index()

st.title('Customer Insights and Product Recommendations')

# -----------------------------------
# Section 1: Time Series Forecasting
st.header('Section 1: Time Series Forecasting')
customer_name = st.selectbox("Select Customer", df['Client Name'].unique())
product_name = st.selectbox("Select Product", df['Product'].unique())

customer_product_df = monthly_purchase_data[(monthly_purchase_data['Client Name'] == customer_name) & 
                                            (monthly_purchase_data['Product'] == product_name)]

prophet_df = customer_product_df[['YearMonth', 'Quantity']]
prophet_df.columns = ['ds', 'y']
prophet_df['ds'] = prophet_df['ds'].dt.to_timestamp()
prophet_df['y'] = pd.to_numeric(prophet_df['y'], errors='coerce')

prophet_df = prophet_df.dropna(subset=['ds', 'y'])

if len(prophet_df) < 2:
   st.write(f"Not enough data for product '{product_name}' and customer '{customer_name}'")
else:
   model = Prophet()
   model.fit(prophet_df)

   future = model.make_future_dataframe(periods=3, freq='M')
   forecast = model.predict(future)

   st.subheader(f"3-Month Forecast for {product_name} for {customer_name}")
   fig, ax = plt.subplots(figsize=(10, 6))
   model.plot(forecast, ax=ax)
   st.pyplot(fig)

# -----------------------------------
# Section 2: Customer Segmentation
st.header('Section 2: Customer Segmentation')
customer_features = df.groupby('Client Name').agg({
    'Quantity': 'sum',
    'Amount': 'sum',
    'Invoice No.': 'nunique'
}).reset_index()
customer_features.columns = ['Client Name', 'Total_Quantity', 'Total_Amount', 'Purchase_Frequency']

fig, ax = plt.subplots(figsize=(10, 6))

sns.scatterplot(data=customer_features, x='Total_Amount', y='Purchase_Frequency', 
                hue='Total_Quantity', palette='viridis', size='Total_Quantity', sizes=(20, 200), legend=None)

plt.title("Customer Segmentation: Total Amount vs Purchase Frequency")
plt.xlabel("Total Amount Spent")
plt.ylabel("Purchase Frequency")
plt.grid(True)
st.pyplot(fig)

# -----------------------------------
# Section 3: Customer Lifetime Value (CLV)
st.header('Section 3: Customer Lifetime Value (CLV) Calculation')
avg_margin = 0.3  
retention_rate = 0.5  
customer_features['CLV'] = (customer_features['Total_Amount'] * avg_margin) / (1 - retention_rate)

st.subheader("Top 10 Customers by Customer Lifetime Value (CLV)")
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(data=customer_features.sort_values('CLV', ascending=False).head(10), x='CLV', y='Client Name', palette='viridis')
plt.title("Top 10 Customers by CLV")
plt.xlabel("Customer Lifetime Value")
plt.ylabel("Client Name")
st.pyplot(fig)

# ---------------------
# **Generate Association Rules with FP-Growth**
basket = df.groupby(['Invoice No.', 'Product'])['Quantity'].sum().unstack().reset_index().set_index('Invoice No.')
basket = basket.applymap(lambda x: 1 if x > 0 else 0)  # Convert quantity to 1 if purchased

frequent_itemsets = fpgrowth(basket, min_support=0.05, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)


# ---------------------
# Section 4: Upsell Recommendations
def get_upsell_recommendations(customer_name, customer_data, product_data, top_n=5):
    customer_purchases = customer_data[customer_data['Client Name'] == customer_name]
    if customer_purchases.empty:
        return f"No purchase history found for customer '{customer_name}'."
    
    max_purchase_price = customer_purchases['Amount'].max()
    upsell_candidates = product_data[product_data['Amount'] > max_purchase_price]
    
    if upsell_candidates.empty:
        return f"No upsell opportunities available for customer '{customer_name}'."
    
    product_popularity = (
        customer_data['Product']
        .value_counts()
        .rename_axis('Product')
        .reset_index(name='Popularity')
    )
    
    upsell_candidates = upsell_candidates.merge(product_popularity, on='Product', how='left').fillna(0)
    upsell_candidates = upsell_candidates.sort_values(by='Amount', ascending=False).drop_duplicates(subset='Product', keep='first')
    upsell_candidates = upsell_candidates.sort_values(by='Popularity', ascending=False).head(top_n)
    
    recommendations = upsell_candidates[['Product', 'Amount', 'Popularity']].to_dict('records')
    
    return recommendations

product_data = customer_product_data[['Product', 'Amount']].drop_duplicates()  # Simplified product data

upsell_recommendations = get_upsell_recommendations(customer_name, customer_product_data, product_data)

st.subheader(f"Top Upsell Recommendations for '{customer_name}':")
for rec in upsell_recommendations:
    st.write(f"Product: {rec['Product']}, Popularity: {rec['Popularity']}")

# ---------------------
# Section 5: Cross-Sell Prediction
def get_cross_sell_recommendations(rules, selected_product, top_n=5):
    selected_product_rules = rules[rules['antecedents'].apply(lambda x: selected_product in x)]
    selected_product_rules = selected_product_rules.sort_values(by='confidence', ascending=False)
    
    unique_recommendations = selected_product_rules.drop_duplicates(subset='consequents', keep='first')
    top_recommendations = unique_recommendations.head(top_n)
    
    recommendations = [
        (list(consequent)[0], confidence) 
        for consequent, confidence in zip(top_recommendations['consequents'], top_recommendations['confidence'])
    ]
    
    return recommendations

recommendations = get_cross_sell_recommendations(rules, product_name)

st.subheader(f"Top Cross-Sell Recommendations for '{product_name}':")
for product, confidence in recommendations:
    st.write(f"{product}: Confidence = {confidence:.2f}")

# -----------------------------------
# Summary
st.header('Summary of Analyses')
st.write("""
1. **Time Series Forecast**: Forecasted future purchase quantities using Prophet.
2. **Customer Segmentation**: Visualized customer engagement based on purchase frequency and total spending.
3. **Customer Lifetime Value (CLV)**: Calculated CLV for each customer to identify valuable customers.
4. **Upsell Prediction**: Suggested products likely to appeal to specific customers based on their purchase history.
5. **Cross-Sell Prediction**: Recommended complementary products based on frequent purchase combinations.
""")

