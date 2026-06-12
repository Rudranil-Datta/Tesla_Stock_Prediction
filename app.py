import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

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
df["Date"] = pd.to_datetime(df["Date"])


# --------------------------------------------------
# PREPROCESS DATA
# --------------------------------------------------

close_prices = df[["Close"]]

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(close_prices)

LOOKBACK = 60

# --------------------------------------------------
# LOAD MODELS
# --------------------------------------------------

@st.cache_resource
def load_models():
    return (
        load_model("models/model_1day.keras"),
        load_model("models/model_5day.keras"),
        load_model("models/model_10day.keras")
    )

model_1, model_5, model_10 = load_models()

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
# METRIC
# --------------------------------------------------

latest_price = float(close_prices["Close"].iloc[-1])

st.metric(
    "Latest Closing Price",
    f"${latest_price:.2f}"
)

# --------------------------------------------------
# FUTURE FORECASTING
# --------------------------------------------------

st.subheader("🔮 Future Forecasts")

last_60_days = scaled_data[-LOOKBACK:]
X_future = np.array([last_60_days])

if st.button("Generate Future Forecasts"):

    with st.spinner("Generating forecasts..."):

        pred_1 = model_1.predict(X_future, verbose=0)
        pred_5 = model_5.predict(X_future, verbose=0)
        pred_10 = model_10.predict(X_future, verbose=0)

        pred_1 = scaler.inverse_transform(pred_1)[0][0]
        pred_5 = scaler.inverse_transform(pred_5)[0][0]
        pred_10 = scaler.inverse_transform(pred_10)[0][0]

        future_df = pd.DataFrame({
            "Forecast Horizon": [
                "1 Day",
                "5 Days",
                "10 Days"
            ],
            "Predicted Price ($)": [
                round(pred_1, 2),
                round(pred_5, 2),
                round(pred_10, 2)
            ]
        })

    st.success("Future forecasts generated successfully")

    st.dataframe(
        future_df,
        width="stretch"
    )

# --------------------------------------------------
# HISTORICAL BACKTESTING
# --------------------------------------------------

st.subheader("📊 Advanced Historical Backtesting")

min_date = df["Date"].iloc[60].date()
max_date = df["Date"].iloc[-11].date()

selected_date = st.date_input(
    "Select Historical Date",
    value=max_date,
    min_value=min_date,
    max_value=max_date
)

selected_date = pd.to_datetime(selected_date)

matches = df[df["Date"].dt.date == selected_date.date()]

if len(matches) == 0:

    nearest_idx = (
        (df["Date"] - selected_date)
        .abs()
        .idxmin()
    )

    nearest_date = df.loc[nearest_idx, "Date"]

    st.warning(
        f"No trading data available for {selected_date.strftime('%Y-%m-%d')}. "
        f"Using nearest trading day: {nearest_date.strftime('%Y-%m-%d')}."
    )

    idx = nearest_idx

else:
    idx = matches.index[0]

historical_window = scaled_data[idx - 60:idx]

X_backtest = np.array([historical_window])

if st.button("Run Historical Backtest"):

    with st.spinner("Generating historical forecasts..."):

        pred_1 = model_1.predict(X_backtest, verbose=0)
        pred_5 = model_5.predict(X_backtest, verbose=0)
        pred_10 = model_10.predict(X_backtest, verbose=0)

        pred_1 = scaler.inverse_transform(pred_1)[0][0]
        pred_5 = scaler.inverse_transform(pred_5)[0][0]
        pred_10 = scaler.inverse_transform(pred_10)[0][0]

        actual_1 = df["Close"].iloc[idx + 1]
        actual_5 = df["Close"].iloc[idx + 5]
        actual_10 = df["Close"].iloc[idx + 10]

        comparison_df = pd.DataFrame({
            "Forecast Horizon": [
                "1 Day",
                "5 Days",
                "10 Days"
            ],
            "Predicted Price ($)": [
                round(pred_1, 2),
                round(pred_5, 2),
                round(pred_10, 2)
            ],
            "Actual Price ($)": [
                round(actual_1, 2),
                round(actual_5, 2),
                round(actual_10, 2)
            ],
            "Absolute Error ($)": [
                round(abs(pred_1 - actual_1), 2),
                round(abs(pred_5 - actual_5), 2),
                round(abs(pred_10 - actual_10), 2)
            ]
        })

    st.success(
        f"Historical forecast generated for {selected_date.strftime('%Y-%m-%d')}"
    )

    st.dataframe(
        comparison_df,
        width="stretch"
    )

# --------------------------------------------------
# HISTORICAL CHART
# --------------------------------------------------

st.subheader("📈 Tesla Historical Closing Prices")

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

st.plotly_chart(
    fig,
    width="stretch"
)

# --------------------------------------------------
# RECENT DATA
# --------------------------------------------------

with st.expander("View Recent Stock Data"):

    st.dataframe(
        df.tail(20),
        width="stretch"
    )

# --------------------------------------------------
# DATASET INFORMATION
# --------------------------------------------------

with st.expander("Dataset Information"):

    st.write(f"Total Records: {len(df)}")

    st.write(
        f"Date Range: {df.iloc[0]['Date'].date()} to {df.iloc[-1]['Date'].date()}"
    )

# --------------------------------------------------
# MODEL INFORMATION
# --------------------------------------------------

with st.expander("Model Information"):

    st.markdown(
        """
        **Model Type:** LSTM

        **Lookback Window:** 60 Days

        **Forecast Horizons**
        - 1 Day Ahead
        - 5 Days Ahead
        - 10 Days Ahead

        **Target Variable:** Closing Price

        **Scaling Method:** MinMaxScaler

        **Additional Feature:** Historical Backtesting
        """
    )

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown("---")

st.caption(
    "Tesla Stock Price Prediction Project | Submitted by Rudranil Datta"
)