import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Recursive Sales/Transactions Forecaster", layout="wide")

@st.cache_resource
def load_models():
    return {
        'sales_model': joblib.load('lightgbm_sales_model_2.11.pkl'),
        'transactions_model': joblib.load('lightgbm_transactions_model_2.11.pkl'),
        'label_encoders': joblib.load('label_encoders.pkl')
    }
models = load_models()

# KEY CHANGE: 'transactions' is needed for sales model input!
sales_features = [
    'store_nbr', 'family', 'onpromotion', 'perishable',
    'oil_price', 'temperature', 'precipitation',
    'day_of_week', 'is_month_start', 'is_month_end',
    'month', 'year', 'store_type', 'city', 'state',
    'day_type', 'Event Scale', 'locale_name',
    'transactions',   # NEW -- used for sales prediction
    'sales_lag_1', 'sales_lag_7', 'sales_lag_14', 'sales_lag_28',
    'sales_roll_mean_7', 'sales_roll_mean_14', 'sales_roll_mean_28',
]
trans_features = [
    'store_nbr', 'family', 'onpromotion', 'perishable',
    'oil_price', 'temperature', 'precipitation',
    'day_of_week', 'is_month_start', 'is_month_end',
    'month', 'year', 'store_type', 'city', 'state',
    'day_type', 'Event Scale', 'locale_name',
    'trans_lag_1', 'trans_lag_7', 'trans_lag_14',
    'trans_roll_mean_7', 'trans_roll_mean_14',
]

def create_date_features(df):
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day_of_week'] = df['date'].dt.dayofweek
    df['is_month_start'] = df['date'].dt.is_month_start.astype(int)
    df['is_month_end'] = df['date'].dt.is_month_end.astype(int)
    return df

def create_lag_rolling_features(df):
    df = df.sort_values(['store_nbr', 'family', 'date'])
    for lag in [1, 7, 14, 28]:
        df[f'sales_lag_{lag}'] = df.groupby(['store_nbr', 'family'])['unit_sales'].shift(lag)
    for lag in [1, 7, 14]:
        df[f'trans_lag_{lag}'] = df.groupby(['store_nbr', 'family'])['transactions'].shift(lag)
    for window in [7, 14, 28]:
        df[f'sales_roll_mean_{window}'] = (
            df.groupby(['store_nbr', 'family'])['unit_sales'].shift(1).rolling(window).mean().reset_index(0, drop=True)
        )
    for window in [7, 14]:
        df[f'trans_roll_mean_{window}'] = (
            df.groupby(['store_nbr', 'family'])['transactions'].shift(1).rolling(window).mean().reset_index(0, drop=True)
        )
    return df

def encode_categoricals(df, label_encoders):
    for col in df.select_dtypes(include=['object', 'string']).columns:
        if col in label_encoders:
            encoder = label_encoders[col]
            df[col] = df[col].map(lambda x: encoder.transform([x])[0] if x in encoder.classes_ else -1)
    return df

def decode_column(df, colname, label_encoder):
    if label_encoder is not None and df[colname].dtype in [np.int64, np.int32, int, np.float64, float]:
        mask = df[colname] != -1
        df.loc[mask, colname+'_decoded'] = label_encoder.inverse_transform(df.loc[mask, colname].astype(int))
        df.loc[~mask, colname+'_decoded'] = 'UNKNOWN'
        return df
    else:
        return df

st.title("Recursive Multi-day Sales/Transactions Forecaster (Chained)")

uploaded_file = st.file_uploader("Upload historical data CSV (w/ all features, up to last known day)", type="csv")
future_dates_instructions = """
Upload a **second CSV** ("prediction plan") listing the rows you wish to forecast:
- Each row should have: `date`, `store_nbr`, `family`, and exogenous/categorical features.
- For all future rows, `unit_sales` and `transactions` can be empty/np.nan; they will be predicted.
- Lag/rolling features will be managed automatically.
"""
st.markdown(f"<div style='background:#eef;padding:10px'>{future_dates_instructions}</div>", unsafe_allow_html=True)
future_file = st.file_uploader("Upload FUTURE plan CSV (dates/models/features to forecast)", type="csv", key="future")

if uploaded_file and future_file:
    hist = pd.read_csv(uploaded_file)
    future = pd.read_csv(future_file)
    hist['date'] = pd.to_datetime(hist['date'])
    future['date'] = pd.to_datetime(future['date'])

    hist = create_date_features(hist)
    future = create_date_features(future)

    needed = ['date', 'store_nbr', 'family', 'onpromotion', 'perishable',
              'oil_price', 'temperature', 'precipitation', 'store_type', 'city', 'state',
              'day_type', 'Event Scale', 'locale_name', 'unit_sales', 'transactions']
    for c in needed:
        if c not in future.columns: future[c] = np.nan
        if c not in hist.columns: hist[c] = np.nan

    results = []
    df_forecast = pd.concat([hist, future[future['date'] > hist['date'].max()]], sort=False)
    df_forecast = df_forecast.sort_values(['store_nbr','family','date']).reset_index(drop=True)

    forecast_dates = sorted(future['date'].unique())
    for forecast_date in forecast_dates:
        mask = (df_forecast['date']==forecast_date) & (df_forecast['unit_sales'].isna())
        if not mask.any(): continue

        # Compute lags/rollings up to this day using history and already predicted rows
        df_forecast = create_lag_rolling_features(df_forecast)

        pred_part = df_forecast[mask].copy()
        for col in set(sales_features + trans_features):
            if col not in pred_part: pred_part[col] = 0
        pred_part[sales_features] = pred_part[sales_features].fillna(0)
        pred_part[trans_features] = pred_part[trans_features].fillna(0)
        pred_part = encode_categoricals(pred_part, models['label_encoders'])

        # KEY CHANGE: First predict transactions, then use them to predict sales
        pred_part['transactions'] = np.expm1(models['transactions_model'].predict(pred_part[trans_features]))
        pred_part['transactions'] = np.clip(pred_part['transactions'], 0, None)
        # Now use predicted transactions as input for sales
        pred_part['unit_sales'] = np.expm1(models['sales_model'].predict(pred_part[sales_features]))
        pred_part['unit_sales'] = np.clip(pred_part['unit_sales'], 0, None)

        results.append(pred_part[["date","store_nbr","family","unit_sales","transactions"]])

        # Update the forecast DataFrame for future lags and rolling
        for i, row in pred_part.iterrows():
            idx = df_forecast[
                (df_forecast['date']==row['date']) &
                (df_forecast['store_nbr']==row['store_nbr']) &
                (df_forecast['family']==row['family'])
            ].index
            if not idx.empty:
                df_forecast.at[idx[0],'unit_sales'] = row['unit_sales']
                df_forecast.at[idx[0],'transactions'] = row['transactions']

    all_preds = pd.concat(results, ignore_index=True) if results else pd.DataFrame()

    # Decode family (optional)
    family_encoder = models['label_encoders'].get('family', None)
    if not all_preds.empty:
        all_preds = decode_column(all_preds, 'family', family_encoder)
        display_cols = ["date", "store_nbr", "family_decoded", "unit_sales", "transactions"] \
            if "family_decoded" in all_preds.columns else ["date", "store_nbr", "family", "unit_sales", "transactions"]
        st.subheader(f"Predicted sales and transactions:")
        st.dataframe(all_preds[display_cols], use_container_width=True)
        st.success("Recursive prediction finished.")
        csv_data = all_preds[display_cols].to_csv(index=False)
        st.download_button(
            label='Download predictions as CSV',
            data=csv_data,
            file_name='predicted_sales_transactions.csv',
            mime='text/csv'
        )
    else:
        st.subheader(f"Predicted sales and transactions:")
        st.write("No prediction rows were generated.")
        st.success("Recursive prediction finished.")

elif uploaded_file:
    st.info("Now upload your FUTURE plan/prediction input CSV.")

else:
    st.info("Upload your historical data first (with all features and actuals up to the latest known day).")