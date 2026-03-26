# streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib
matplotlib.use('Agg') # This forces matplotlib to run without a GUI
import matplotlib.pyplot as plt

# ==========================================
# 1. Setup the Page Layout
# ==========================================
st.set_page_config(page_title="Indian Stock Predictor", layout="wide")
st.title("📈 AI Stock Market Predictor")

# ==========================================
# 2. Dynamic File Scanner (Kaggle Dataset)
# ==========================================
DATA_DIR = "data/"
if os.path.exists(DATA_DIR):
    stock_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith('.csv')])
else:
    stock_files = []

if not stock_files:
    st.error("No data files found! Please put your Kaggle stock CSV files in the 'data/' folder.")
    st.stop()

# ==========================================
# 3. Sidebar for User Input
# ==========================================
st.sidebar.header("Dashboard Controls")
selected_stock = st.sidebar.selectbox("Select a Stock to Analyze", stock_files)
ticker_name = selected_stock.replace('.csv', '')

# ==========================================
# 4. Data Loading & Feature Engineering
# ==========================================
@st.cache_data
def load_and_clean_data(filename):
    file_path = os.path.join(DATA_DIR, filename)
    df = pd.read_csv(file_path)
    
    # Ensure Date is the index and sorted
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').set_index('Date')
    df = df.ffill().dropna()
    
    # 1. Existing Features
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    df['Daily_Return'] = df['Close'].pct_change()
    
    # 2. Bollinger Bands
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['Std_Dev_20'] = df['Close'].rolling(window=20).std()
    df['Upper_Band'] = df['SMA_20'] + (df['Std_Dev_20'] * 2)
    df['Lower_Band'] = df['SMA_20'] - (df['Std_Dev_20'] * 2)
    
    # 3. NEW: Relative Strength Index (RSI - 14 Day)
    delta = df['Close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ema_up = up.ewm(com=13, adjust=False).mean()
    ema_down = down.ewm(com=13, adjust=False).mean()
    rs = ema_up / ema_down
    df['RSI_14'] = 100 - (100 / (1 + rs))
    
    return df.dropna()

df = load_and_clean_data(selected_stock)

# Extract the "Next Day" for our UI dynamically based on the CSV data
last_date_in_csv = df.index[-1]
target_prediction_date = last_date_in_csv + pd.Timedelta(days=1)
formatted_target_date = target_prediction_date.strftime("%B %d, %Y")

# ==========================================
# 5. Build the UI using Tabs
# ==========================================
st.subheader(f"Dashboard: {ticker_name}")

tab1, tab2, tab3 = st.tabs(["📈 Interactive Chart", "📊 Raw Data & Stats", "🧠 Model Details"])

with tab1:
    # 1. Added subplot_titles so the user knows what they are looking at!
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.08, row_heights=[0.8, 0.2],
                        subplot_titles=(f"{ticker_name} Price Action & Indicators", "Trading Volume"))

    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], line=dict(color='#3498db', width=1.5), name='50 SMA'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Upper_Band'], line=dict(color='rgba(255, 255, 255, 0.3)', width=1, dash='dot'), name='Upper Band'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Lower_Band'], line=dict(color='rgba(255, 255, 255, 0.3)', width=1, dash='dot'), fill='tonexty', fillcolor='rgba(255, 255, 255, 0.05)', name='Lower Band'), row=1, col=1)

    volume_colors = ['rgba(39, 174, 96, 0.7)' if close_price >= open_price else 'rgba(231, 76, 60, 0.7)' 
                     for close_price, open_price in zip(df['Close'], df['Open'])]
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color=volume_colors, name='Volume'), row=2, col=1)

    # 2. Added explicit Y-Axis labels
    fig.update_yaxes(title_text="Price (₹)", row=1, col=1)
    fig.update_yaxes(title_text="Volume Traded", row=2, col=1)

# 3. Moved the legend to the BOTTOM CENTER for a perfectly symmetrical look
    fig.update_layout(
        template='plotly_dark', 
        xaxis_rangeslider_visible=False, 
        xaxis2_rangeslider_visible=False,
        height=680, # Made the chart slightly taller to accommodate the legend
        margin=dict(l=0, r=0, t=50, b=60), # Added 60px of space at the bottom (b=60)
        showlegend=True,
        legend=dict(
            orientation="h", 
            yanchor="top", 
            y=-0.15, # Pushes it below the Volume chart
            xanchor="center", 
            x=0.5    # Perfectly centers it horizontally
        ) 
    )
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    col1, col2 = st.columns([2, 1])
    with col1:
        st.dataframe(df.tail(10), use_container_width=True) 
    with col2:
        st.write("### Current Technicals")
        st.metric(label="Latest Close Price", value=f"₹{df['Close'].iloc[-1]:.2f}")
        st.metric(label="50-Day SMA", value=f"₹{df['SMA_50'].iloc[-1]:.2f}")
        
        current_rsi = df['RSI_14'].iloc[-1]
        rsi_status = "Neutral"
        if current_rsi > 70: rsi_status = "🔴 Overbought"
        elif current_rsi < 30: rsi_status = "🟢 Oversold"
        st.metric(label=f"14-Day RSI ({rsi_status})", value=f"{current_rsi:.1f}")

    st.markdown("---")
    
    rsi_fig = go.Figure()
    rsi_fig.add_trace(go.Scatter(x=df.index, y=df['RSI_14'], line=dict(color='#9b59b6', width=2), name='RSI'))
    
    rsi_fig.add_hline(y=70, line_dash="dash", line_color="rgba(231, 76, 60, 0.8)", annotation_text="Overbought (70)", annotation_position="top left")
    rsi_fig.add_hline(y=30, line_dash="dash", line_color="rgba(39, 174, 96, 0.8)", annotation_text="Oversold (30)", annotation_position="bottom left")

# Centered the legend at the bottom for consistency with Tab 1
    rsi_fig.update_layout(
        title_text="Relative Strength Index (14-Day)",
        template='plotly_dark',
        height=350, # Made slightly taller (from 300 to 350) to fit the legend
        margin=dict(l=0, r=0, t=40, b=60), # Added bottom margin (b=60)
        yaxis=dict(range=[0, 100], title_text="RSI Value"),
        showlegend=True,
        legend=dict(
            orientation="h", 
            yanchor="top", 
            y=-0.25, # Pushed below the X-axis of the RSI chart
            xanchor="center", 
            x=0.5    # Perfectly centered horizontally
        )
    )
    st.plotly_chart(rsi_fig, use_container_width=True)

with tab3:
    st.write("### How this App Works")
    st.info("This application uses Machine Learning to forecast stock prices using historical Kaggle data. If a dedicated Deep Learning (LSTM) model has been trained for the selected stock, it will be used for high-accuracy predictions. Otherwise, a fast Random Forest algorithm will be trained on the fly to provide a baseline estimate.")

# ==========================================
# 6. THE AI PREDICTION ENGINE
# ==========================================
st.markdown("---")
# NEW: Dynamic Date Header!
st.subheader(f"🤖 AI Price Prediction for: {formatted_target_date}")

expected_model_path = f"models/{ticker_name}_lstm.keras"

if os.path.exists(expected_model_path):
    st.success(f"🧠 Expert Deep Learning Model found for {ticker_name}!")
    
    if st.button(f"Predict Close for {formatted_target_date}"):
        with st.spinner('Neural Network is analyzing the sequences...'):
            try:
                model = load_model(expected_model_path)
                
                # We intentionally DO NOT pass RSI into the model here, because your 
                # previously saved LSTM models weren't trained with it. 
                features = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_50', 'SMA_200', 'Daily_Return']
                
                recent_data = df.tail(100)
                feature_scaler = MinMaxScaler()
                scaled_features = feature_scaler.fit_transform(recent_data[features])
                
                target_scaler = MinMaxScaler()
                target_scaler.fit(recent_data[['Close']]) 

                # 1. Predict Target Date
                last_60_scaled = scaled_features[-60:]
                X_input = np.array([last_60_scaled])
                prediction_scaled = model.predict(X_input)
                predicted_price = target_scaler.inverse_transform(prediction_scaled)[0][0]
                
                # 2. BACKTESTING LOGIC
                backtest_days = 30
                X_backtest = []
                for i in range(backtest_days):
                    start_idx = len(scaled_features) - 60 - backtest_days + i
                    end_idx = start_idx + 60
                    X_backtest.append(scaled_features[start_idx:end_idx])
                
                backtest_preds_scaled = model.predict(np.array(X_backtest))
                backtest_preds = target_scaler.inverse_transform(backtest_preds_scaled).flatten()
                
                actuals = recent_data['Close'].tail(backtest_days).values
                backtest_df = pd.DataFrame({
                    'Actual Price': actuals, 
                    'AI Predicted Price': backtest_preds
                }, index=recent_data.tail(backtest_days).index)

                # 3. UI UPGRADE
                latest_actual_price = df['Close'].iloc[-1]
                expected_change = predicted_price - latest_actual_price
                percentage_change = (expected_change / latest_actual_price) * 100
                
                st.metric(label="Predicted Closing Price (LSTM Expert)", value=f"₹{predicted_price:.2f}", delta=f"₹{expected_change:.2f} ({percentage_change:.2f}%)")
                
                st.write("### 📈 Visual Backtest (Last 30 Days)")
                st.line_chart(backtest_df, color=["#2ecc71", "#e74c3c"])
                
                with st.expander("Show AI Confidence & Disclaimers"):
                    st.write("**Model Used:** Deep Learning LSTM")
                    st.caption(f"*Disclaimer: This prediction is generated by an advanced LSTM model explicitly trained on the historical sequence of {ticker_name}.*")
                
            except Exception as e:
                st.error(f"Error making LSTM prediction: {e}")

else:
    # THE FALLBACK MECHANISM (WITH LEAKAGE FIX)
    st.warning(f"⚠️ No Expert Model found for {ticker_name}. We will use a fast Fallback Algorithm.")
    
    if st.button(f"Generate Quick Prediction for {formatted_target_date}"):
        with st.spinner('Training a lightweight Random Forest model on the fly...'):
            try:
                rf_df = df.copy()
                rf_df['Target_Next_Close'] = rf_df['Close'].shift(-1)
                rf_df = rf_df.dropna()
                
                features = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_50', 'SMA_200', 'Daily_Return']
                X_all = rf_df[features]
                y_all = rf_df['Target_Next_Close']
                
                # --- NEW: THE DATA LEAKAGE FIX (Train/Test Split) ---
                backtest_days = 30
                X_train = X_all.iloc[:-backtest_days]
                y_train = y_all.iloc[:-backtest_days]
                X_test = X_all.iloc[-backtest_days:]
                y_test = y_all.iloc[-backtest_days:]
                
                # Train the Random Forest strictly on the past to test its accuracy
                rf_model_blind = RandomForestRegressor(n_estimators=100, random_state=42)
                rf_model_blind.fit(X_train, y_train)
                test_predictions = rf_model_blind.predict(X_test)
                model_mae = mean_absolute_error(y_test, test_predictions)
                
                # Retrain on ALL data to give the best possible future guess
                rf_model_final = RandomForestRegressor(n_estimators=100, random_state=42)
                rf_model_final.fit(X_all, y_all)
                
                today_data = pd.DataFrame([df[features].iloc[-1].values], columns=features)
                predicted_price = rf_model_final.predict(today_data)[0]
                
                # HONEST BACKTESTING LOGIC 
                backtest_df = pd.DataFrame({
                    'Actual Price': y_test.values,
                    'AI Predicted Price (Blind Tested)': test_predictions
                }, index=y_test.index)

                # UI UPGRADE
                latest_actual_price = df['Close'].iloc[-1]
                expected_change = predicted_price - latest_actual_price
                percentage_change = (expected_change / latest_actual_price) * 100
                
                st.metric(label="Predicted Closing Price (Random Forest)", value=f"₹{predicted_price:.2f}", delta=f"₹{expected_change:.2f} ({percentage_change:.2f}%)")
                
                st.write("### 📈 Visual Backtest (Honest Last 30 Days)")
                st.line_chart(backtest_df, color=["#2ecc71", "#e74c3c"])
                
                with st.expander("Show AI Confidence & Disclaimers"):
                    st.write("**Model Used:** Fallback Random Forest (with Train/Test Split)")
                    st.write(f"📊 **Honest Historical Error (MAE):** On data it has never seen before, this model's predictions deviate by an average of **₹{model_mae:.2f}** from the actual price.")
                    st.caption(f"*Disclaimer: This prediction was generated using a lightweight Random Forest algorithm.*")
                
            except Exception as e:
                st.error(f"Error making Fallback prediction: {e}")