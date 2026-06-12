# 🚗 Tesla Stock Price Prediction using Deep Learning

## Overview

This project focuses on forecasting Tesla (TSLA) stock closing prices using Deep Learning techniques. Historical stock market data is processed and used to train Recurrent Neural Network (RNN) and Long Short-Term Memory (LSTM) models capable of learning temporal patterns in stock prices.

The project provides forecasts for:

* 1-Day Ahead Prediction
* 5-Day Ahead Prediction
* 10-Day Ahead Prediction

A Streamlit web application is included to allow users to interactively generate forecasts and visualize Tesla's historical stock performance.

---

## Project Objectives

* Analyze historical Tesla stock price data.
* Perform data preprocessing and feature engineering.
* Build and compare RNN and LSTM models.
* Forecast future stock prices for multiple prediction horizons.
* Evaluate model performance using regression metrics.
* Deploy an interactive prediction dashboard using Streamlit.

---

## Tech Stack

### Programming Language

* Python 3.10

### Libraries and Frameworks

* TensorFlow 2.16.1
* Pandas
* NumPy
* Scikit-Learn
* Matplotlib
* Plotly
* Streamlit
* Jupyter Notebook

---

## Project Structure

```text
Tesla_Stock_Prediction/
│
├── app.py
├── README.md
├── requirements.txt
├── TSLA.csv
|-- .gitignore
│
├── models/
│   ├── simple_rnn_model.keras
│   ├── best_rnn_model.keras
│   ├── lstm_model.keras
│   ├── best_lstm_model.keras
│   ├── final_lstm_model.keras
│   ├── model_1day.keras
│   ├── model_5day.keras
│   └── model_10day.keras
│
├── notebook/
│   └── Tesla_Stock_Prediction.ipynb
│
└── venv/

```

### Folder Description

| Item             | Description                              |
| ---------------- | ---------------------------------------- |
| app.py           | Streamlit application                    |
| TSLA.csv         | Historical Tesla stock dataset           |
| models/          | Saved trained deep learning models       |
| notebook/        | Development and experimentation notebook |
| requirements.txt | Project dependencies                     |
| README.md        | Project documentation                    |

---

## Dataset

The project uses historical Tesla stock market data containing:

* Date
* Open Price
* High Price
* Low Price
* Close Price
* Volume

Target Variable:

```text
Close Price
```

---

## Data Preprocessing

The following preprocessing steps were performed:

1. Loading historical stock data.
2. Date parsing and sorting.
3. Selection of closing prices.
4. Feature scaling using MinMaxScaler.
5. Creation of rolling sequences using a lookback window of 60 days.
6. Train-test splitting.

---

## Model Architecture

### Simple RNN

Used as a baseline deep learning model for sequence forecasting.

### LSTM

Chosen as the final forecasting model due to its ability to capture long-term dependencies in time-series data.

Model Features:

* Multiple recurrent layers
* Dropout regularization
* Early stopping
* Model checkpointing

---

## Forecast Horizons

Three separate forecasting models were trained:

| Model             | Forecast Horizon               |
| ----------------- | ------------------------------ |
| model_1day.keras  | Predicts next trading day      |
| model_5day.keras  | Predicts 5 trading days ahead  |
| model_10day.keras | Predicts 10 trading days ahead |

| Model              | Description           |
| ------------------ | --------------------- |
| `simple_rnn_model` | Baseline RNN model    |
| `best_rnn_model`   | Best RNN checkpoint   |
| `lstm_model`       | Initial LSTM model    |
| `best_lstm_model`  | Best LSTM checkpoint  |
| `final_lstm_model` | Final optimized LSTM  |
| `model_1day`       | 1-day forecast model  |
| `model_5day`       | 5-day forecast model  |
| `model_10day`      | 10-day forecast model |

---

## Live link : https://teslastockprediction-lhjmvfpekwdxerqqyb4oj8.streamlit.app/

## Running the Project locally

### Step 1: Clone Repository

```bash
git clone https://github.com/Rudranil-Datta/Tesla_Stock_Prediction.git
cd Tesla_Stock_Prediction
```

### Step 2: Create Virtual Environment

The project was developed using:

```text
Python 3.10
```

Create a virtual environment:

```bash
python3.10 -m venv venv
```

Activate:

#### macOS / Linux

```bash
source venv/bin/activate
```

#### Windows

```bash
venv\Scripts\activate
```

---

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

---

### Step 4: Launch Streamlit Application

```bash
streamlit run app.py
```

The application will be available at:

```text
http://localhost:8501
```

---

## Streamlit Features

The dashboard provides:

* Historical Tesla stock price visualization
* Forecast horizon selection
* 1-Day prediction
* 5-Day prediction
* 10-Day prediction
* Interactive charts
* Dataset preview
* Model information

---

## Model Evaluation

Performance was evaluated using regression metrics such as:

* RMSE (Root Mean Squared Error)

The forecasting performance across different horizons was compared to understand how prediction accuracy changes as the forecast period increases.

---

## Key Learnings

This project demonstrates:

* Time-series forecasting using deep learning
* Sequence generation techniques
* RNN and LSTM architectures
* Hyperparameter tuning
* Model deployment with Streamlit
* End-to-end machine learning workflow

---

## Environment Information

### Python Version

```text
Python 3.10
```

### TensorFlow Version

```text
TensorFlow 2.16.1
```

Note:

The development machine may contain a global Python 3.14 installation. However, this project was executed inside an isolated virtual environment using Python 3.10 for TensorFlow compatibility and reproducibility.

---

## Future Improvements

* Incorporate technical indicators (RSI, MACD, Moving Averages)
* Add real-time stock data integration
* Experiment with GRU and Transformer models
* Multi-feature forecasting
* Automated retraining pipeline
* Cloud deployment

---

## Author

Internship Project by Rudranil Datta

Tesla Stock Price Prediction using Deep Learning and Streamlit.
