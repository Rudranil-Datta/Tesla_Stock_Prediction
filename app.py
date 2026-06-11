import streamlit as st
import pandas as pd
import numpy as np

from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Tesla Stock Price Prediction",
    page_icon="🚗",
    layout="wide"
)

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

@st.cache_data
def load_data():
    return pd.read_csv("TSLA.csv")

df = load_data()

# --------------------------------------------------
# PREPROCESS DATA
# --------------------------------------------------

close_prices = df[['Close']]

scaler = MinMaxScaler(feature_range=(0,1))

scaled_data = scaler.fit_transform(close_prices)

LOOKBACK = 60

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.title("Forecast Settings")

horizon = st.sidebar.selectbox(
    "Select Forecast Horizon",
    [
        "1 Day",
        "5 Days",
        "10 Days"
    ]
)

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

@st.cache_resource
def load_prediction_model(horizon):

    if horizon == "1 Day":
        return load_model("models/model_1day.keras")

    elif horizon == "5 Days":
        return load_model("models/model_5day.keras")

    else:
        return load_model("models/model_10day.keras")


model = load_prediction_model(horizon)

# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("🚗 Tesla Stock Price Prediction")

st.markdown(
    """
    Deep Learning based Tesla Stock Forecasting using LSTM Networks.
    """
)

# --------------------------------------------------
# METRICS
# --------------------------------------------------

latest_price = float(close_prices.iloc[-1])

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Latest Closing Price",
        f"${latest_price:.2f}"
    )

with col2:
    st.metric(
        "Forecast Horizon",
        horizon
    )

# --------------------------------------------------
# PREPARE INPUT
# --------------------------------------------------

last_60_days = scaled_data[-LOOKBACK:]

X_input = np.array([last_60_days])

# --------------------------------------------------
# PREDICTION BUTTON
# --------------------------------------------------

if st.button("Generate Forecast"):



    with st.spinner("Generating forecast..."):
    
        prediction = model.predict(
            X_input,
            verbose=0
        )


        predicted_price = scaler.inverse_transform(
            prediction
        )

    st.success(
        f"Predicted Tesla Closing Price: ${predicted_price[0][0]:.2f}"
    )


    



# --------------------------------------------------
# HISTORICAL CHART
# --------------------------------------------------

st.subheader("Tesla Historical Closing Prices")

df['Date'] = pd.to_datetime(df['Date'])

chart_df = df[['Date', 'Close']].copy()
chart_df = chart_df.set_index('Date')

#st.line_chart(chart_df)

import plotly.express as px

fig = px.line(
    df,
    x="Date",
    y="Close",
    title="Tesla Stock Price History",
    labels={
        "Date": "Year",
        "Close": "Closing Price (USD $)"
    }
)

st.plotly_chart(fig, width="stretch")

# --------------------------------------------------
# RECENT DATA
# --------------------------------------------------

with st.expander("View Recent Stock Data"):

    st.dataframe(
        df.tail(20),
        use_container_width=True
    )

# --------------------------------------------------
# DATASET INFORMATION
# --------------------------------------------------

with st.expander("Dataset Information"):

    st.write(
        f"Total Records: {len(df)}"
    )

    st.write(
        f"Date Range: {df.iloc[0]['Date']} to {df.iloc[-1]['Date']}"
    )

# --------------------------------------------------
# MODEL INFORMATION
# --------------------------------------------------

with st.expander("Model Information"):

    st.markdown(
        """
        **Model Type:** LSTM

        **Lookback Window:** 60 Days

        **Forecast Horizons:**
        - 1 Day Ahead
        - 5 Days Ahead
        - 10 Days Ahead

        **Target Variable:** Closing Price

        **Scaling Method:** MinMaxScaler
        """
    )

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown("---")

st.caption(
    "Tesla Stock Price Prediction Project | Submitted by Rudranil Datta"
)