from flask import Flask, request, jsonify
import pandas as pd
import joblib
import numpy as np
from datetime import datetime

app = Flask(__name__)

# Load models and encoders
sales_model = joblib.load('lightgbm_sales_model_2.11.pkl')
transactions_model = joblib.load('lightgbm_transactions_model_2.11.pkl')
label_encoders = joblib.load('label_encoders.pkl')

# Required feature lists
sales_model_features = [
    'store_nbr', 'family', 'onpromotion', 'perishable',
    'oil_price', 'temperature', 'precipitation',
    'day_of_week', 'is_month_start', 'is_month_end',
    'month', 'year', 'store_type', 'city', 'state',
    'day_type', 'Event Scale', 'locale_name',
    'sales_lag_1', 'sales_lag_7', 'sales_lag_14', 'sales_lag_28',
    'sales_roll_mean_7', 'sales_roll_mean_14', 'sales_roll_mean_28',
]

transactions_model_features = [
    'store_nbr', 'family', 'onpromotion', 'perishable',
    'oil_price', 'temperature', 'precipitation',
    'day_of_week', 'is_month_start', 'is_month_end',
    'month', 'year', 'store_type', 'city', 'state',
    'day_type', 'Event Scale', 'locale_name',
    'trans_lag_1', 'trans_lag_7', 'trans_lag_14',
    'trans_roll_mean_7', 'trans_roll_mean_14',
]

# Feature engineering functions
def create_date_features(df):
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['day_of_week'] = df['date'].dt.dayofweek
    df['quarter'] = df['date'].dt.quarter
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['is_month_start'] = df['date'].dt.is_month_start.astype(int)
    df['is_month_end'] = df['date'].dt.is_month_end.astype(int)
    df['weekofyear'] = df['date'].dt.isocalendar().week.astype(int)
    return df

def create_lag_features(df):
    df = df.sort_values(['store_nbr', 'family', 'date'])
    for lag in [1, 7, 14, 28]:
        df[f'sales_lag_{lag}'] = df.groupby(['store_nbr', 'family'])['unit_sales'].shift(lag)
    for lag in [1, 7, 14]:
        df[f'trans_lag_{lag}'] = df.groupby('store_nbr')['transactions'].shift(lag)
    return df

def create_rolling_features(df):
    for window in [7, 14, 28]:
        df[f'sales_roll_mean_{window}'] = (
            df.groupby(['store_nbr', 'family'])['unit_sales'].shift(1).rolling(window).mean()
        )
    for window in [7, 14]:
        df[f'trans_roll_mean_{window}'] = (
            df.groupby('store_nbr')['transactions'].shift(1).rolling(window).mean()
        )
    return df

def create_group_features(df):
    df['mean_transactions_family'] = df.groupby('family')['transactions'].transform('mean')
    df['mean_transactions_store'] = df.groupby('store_nbr')['transactions'].transform('mean')
    df['mean_transactions_store_family'] = df.groupby(['store_nbr', 'family'])['transactions'].transform('mean')
    return df

def handle_outliers(df):
    numeric_cols = ['unit_sales', 'transactions']
    for col in numeric_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - (1.5 * iqr)
        upper_bound = q3 + (1.5 * iqr)
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    return df

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No input data provided"}), 400

        df = pd.DataFrame(data)
        if df.empty:
            return jsonify({"error": "Empty data frame received"}), 400

        df = create_date_features(df)
        df = handle_outliers(df)
        df = create_lag_features(df)
        df = create_rolling_features(df)
        df = create_group_features(df)

        required_features = [
            'sales_lag_1', 'sales_lag_7', 'sales_roll_mean_7',
            'trans_lag_1', 'trans_roll_mean_7'
        ]
        ''''df.dropna(subset=required_features, inplace=True)
        if df.empty:
            return jsonify({"error": "Insufficient historical data for prediction"}), 400'''

        # Encode categorical features
        for col in df.select_dtypes(include=['object', 'string']).columns:
            if col in label_encoders:
                df[col] = label_encoders[col].transform(df[col])

        # Final feature sets for prediction
        X_sales = df[sales_model_features]
        X_trans = df[transactions_model_features]

        sales_pred = sales_model.predict(X_sales)
        transactions_pred = transactions_model.predict(X_trans)

        return jsonify({
            "predictions": {
                "sales": sales_pred.tolist(),
                "transactions": transactions_pred.tolist(),
                "timestamp": datetime.now().isoformat()
            }
        })

    except Exception as e:
        app.logger.error(f"Prediction error: {str(e)}")
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

if __name__ == '__main__':
    import os
    app.run(debug=True, use_reloader=False)

