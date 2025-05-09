import streamlit as st
import pandas as pd
import plotly.express as px

# Load data
@st.cache_data
def load_data():
    return pd.read_parquet(r"C:\Users\alqay\OneDrive\Desktop\store-sales-time-series-forecasting", engine="pyarrow")

df = load_data()
df['date'] = pd.to_datetime(df['date'])

# Title and description
st.title("Store Sales Dashboard")
st.markdown("Interactive EDA Dashboard for Store Sales Time Series Forecasting")

# Sidebar filters
st.sidebar.header("Filter Options")
city = st.sidebar.selectbox("Select City", df['city'].unique())
store_type = st.sidebar.selectbox("Select Store Type", df['store_type'].unique())
date_range = st.sidebar.date_input("Date Range", [df['date'].min(), df['date'].max()])

# Apply filters
filtered_df = df[
    (df['city'] == city) &
    (df['store_type'] == store_type) &
    (df['date'].between(date_range[0], date_range[1]))
]

# 1. Sales Over Time
st.subheader("Unit Sales Over Time")
fig1 = px.line(filtered_df, x="date", y="unit_sales", color="family", title="Unit Sales by Family")
st.plotly_chart(fig1, use_container_width=True)

# 2. Sales Distribution
st.subheader("Unit Sales Distribution")
fig2 = px.histogram(filtered_df, x="unit_sales", nbins=50, title="Sales Distribution")
st.plotly_chart(fig2, use_container_width=True)

# 3. Promotions Impact
st.subheader("Promotions vs. Sales")
fig3 = px.box(filtered_df, x="onpromotion", y="unit_sales", points="all", title="Effect of Promotion on Sales")
st.plotly_chart(fig3, use_container_width=True)

# 4. Temperature vs. Sales
st.subheader("Temperature vs. Unit Sales")
fig4 = px.scatter(filtered_df, x="temperature", y="unit_sales", trendline="ols", title="Temperature vs. Sales")
st.plotly_chart(fig4, use_container_width=True)

# 5. Day Type Impact
st.subheader("Day Type vs. Unit Sales")
fig5 = px.box(filtered_df, x="day_type", y="unit_sales", color="day_type", title="Sales by Day Type")
st.plotly_chart(fig5, use_container_width=True)

