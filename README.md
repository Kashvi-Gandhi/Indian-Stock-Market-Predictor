# AI Stock Market Predictor

> An end-to-end, interactive web application built with Streamlit that leverages Machine Learning to forecast Indian stock market prices. This system features a dynamic dual-engine AI backend, advanced feature engineering, and a professional-grade technical analysis dashboard.

---

## Core Capabilities

* **Dual-Engine AI Backend** * **Deep Learning (LSTM):** Executes high-accuracy, sequence-based forecasting by loading pre-trained `.keras` neural networks.
  * **Machine Learning Fallback:** Dynamically trains a Random Forest Regressor on-the-fly if a dedicated LSTM model is unavailable for the selected asset.
* **Leak-Proof Backtesting**
  * Implements strict sequential Train/Test splitting to completely eliminate data leakage. The models are evaluated purely on the last 30 days of *unseen* data to provide an honest Mean Absolute Error (MAE) and visual performance backtest.
* **Advanced Feature Engineering**
  * Automatically calculates and visualizes core technical indicators from raw historical data:
    * 50-Day & 200-Day Simple Moving Averages (SMA)
    * Bollinger Bands (Volatility tracking)
    * 14-Day Relative Strength Index (RSI) with dynamic Overbought/Oversold flagging
* **Professional UI/UX**
  * Built with Plotly, featuring interactive candlestick charts, dynamic layout spacing, centralized legends, and clear technical summaries.

---

## Technology Stack

* **Frontend:** Streamlit, Plotly Graph Objects
* **Machine Learning:** TensorFlow/Keras (LSTM), Scikit-Learn (Random Forest)
* **Data Processing:** Pandas, NumPy, Scikit-Learn (MinMaxScaler)

---

## Application Interface Preview

*[Insert a high-resolution screenshot of your Streamlit dashboard here, showcasing the interactive charts and RSI metrics]*

---

## Project Architecture

```text
ai-stock-predictor/
├── app.py                  # Main Streamlit application and UI routing
├── data/                   # Directory containing historical Kaggle CSV datasets
├── models/                 # Directory containing pre-trained LSTM (.keras) files
├── README.md               # System documentation
└── requirements.txt        # Python dependency specifications