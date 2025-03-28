# terminal
# python3 -m streamlit run univar_predict.py

import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tools.sm_exceptions import ConvergenceWarning
import matplotlib.pyplot as plt
import warnings
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
import random
import warnings

# Suppress warnings and apply visual style
warnings.filterwarnings("ignore", category=ConvergenceWarning)
plt.style.use('fivethirtyeight')

st.set_page_config(page_title="Univariate Forecast App - SARIMA", layout="wide")
st.title("📈 Univariate Forecast App")

st.markdown("""
This app uses a **univariate time series model** (ARIMA) to forecast future values based only on historical trends of a single variable.  
Why use a univariate model instead of simply comparing to the same period last year?

- It can **capture seasonality, trend, and noise** in the data more flexibly.
- Unlike "last year" comparisons, univariate models can **adapt to shifting baselines** or changing behavior over time.
- You don’t need external variables — this makes it easier to deploy with limited data.

Upload your data below and generate a forward-looking forecast.
""")

# Upload file
uploaded_file = st.file_uploader("Upload your Excel or CSV file", type=["xlsx", "csv"])

if uploaded_file:
    # Read data
    if uploaded_file.name.endswith('.xlsx'):
        df = pd.read_excel(uploaded_file)
    else:
        df = pd.read_csv(uploaded_file)
    
    st.subheader("📊 Raw Data")
    st.write(df.head())

    # Preprocess
    df['date'] = pd.to_datetime(df['date'])  # Keep full datetime
    df.set_index('date', inplace=True)

    # Forecast settings
    st.sidebar.subheader("🔮 Forecast Settings")
    pred_steps = st.sidebar.number_input("Steps to Forecast", min_value=1, max_value=52, value=6, step=1)

    # Train/test split
    train = df.iloc[:-pred_steps]
    test = df.iloc[-pred_steps:]


    # Define parameter ranges
    p_values = range(0, 3)
    d_values = range(0, 3)
    q_values = range(0, 3)
    P_values = range(0, 3)
    D_values = range(0, 3)
    Q_values = range(0, 3)
    #m = 5  # seasonal period

    # Let the user set the seasonal value "m"
    m = st.number_input(
        label="Set seasonal value (m)",
        min_value=1,
        max_value=52,
        value=5,
        help="Number of seasonal periods in your data. For example, use 12 for monthly data with yearly seasonality."
    )

    # Generate random combinations of parameters
    param_combinations = [(p, d, q, P, D, Q) 
                        for p in p_values 
                        for d in d_values 
                        for q in q_values 
                        for P in P_values 
                        for D in D_values 
                        for Q in Q_values]

    random.seed(42)
    random.shuffle(param_combinations)
    sample_combinations = param_combinations[:25]  # sample 25 random combinations

    best_aic = np.inf
    best_order = None
    best_seasonal_order = None
    best_model = None

    for (p, d, q, P, D, Q) in sample_combinations:
        try:
            model = SARIMAX(train, order=(p, d, q), seasonal_order=(P, D, Q, m)).fit(disp=False)
            if model.aic < best_aic:
                best_aic = model.aic
                best_order = (p, d, q)
                best_seasonal_order = (P, D, Q, m)
                best_model = model
        except:
            continue

    print(f"Best SARIMA order: {best_order}")
    print(f"Best seasonal order: {best_seasonal_order}")

    # Forecast
    pred = best_model.get_prediction(start=len(train), end=len(train)+len(test)-1, dynamic=False)
    forecast = pred.predicted_mean
    conf_int = pred.conf_int()

    # Calculate error
    error = mean_squared_error(test, forecast)
    print(f'Mean Squared Error: {error:.2f}')

    forecast.index = test.index  # Ensure alignment

    # Plot controls
    st.sidebar.subheader("🛠️ Plot Settings")
    plot_title = st.sidebar.text_input("Plot Title", "Forecast vs Actuals")
    y_label = st.sidebar.text_input("Y-axis Label", df.columns[0])
    show_forecast_error = st.sidebar.checkbox("Show Forecast Error Range", value=True)

    # Date filter
    st.sidebar.subheader("📅 Plot Filters")
    min_date = df.index.min()
    max_date = df.index.max()

    plot_start_date = st.sidebar.date_input(
        "Start Plot From This Date",
        value=min_date,
        min_value=min_date,
        max_value=max_date
    )
    plot_start_date = pd.to_datetime(plot_start_date)

    # Filter data
    plot_train = train[train.index >= plot_start_date]
    plot_test = test[test.index >= plot_start_date]
    plot_forecast = forecast[forecast.index >= plot_start_date]
    plot_conf = conf_int[conf_int.index >= plot_start_date]

    # Plot
    st.subheader("📉 Forecast Plot")
    fig, ax = plt.subplots(figsize=(10, 5))

    if not plot_train.empty:
        ax.plot(plot_train.index, plot_train.iloc[:, 0], label="Actual Data", color='grey', linewidth=2)

    if not plot_test.empty:
        ax.plot(plot_test.index, plot_test.iloc[:, 0], color='grey', linewidth=2)

    if not plot_forecast.empty:
        ax.plot(plot_forecast.index, plot_forecast, label="Forecast", color='red', linestyle='--', linewidth=2)

        if show_forecast_error and plot_conf is not None:
            ax.fill_between(
                plot_conf.index,
                plot_conf.iloc[:, 0],
                plot_conf.iloc[:, 1],
                color='red',
                alpha=0.2,
                label="Confidence Interval"
            )

    ax.set_title(plot_title, fontsize=16)
    ax.set_ylabel(y_label, fontsize=12)
    ax.grid(alpha=0.3)
    ax.legend()
    st.pyplot(fig)

    # Show predictions
    result_df = pd.DataFrame({
        'Actual': test.iloc[:, 0],
        'Forecast': forecast
    })

    st.subheader("📋 Predictions")
    st.dataframe(result_df)

    # Download option
    csv = result_df.to_csv().encode('utf-8')
    st.download_button("Download Predictions as CSV", csv, "forecast.csv", "text/csv")
