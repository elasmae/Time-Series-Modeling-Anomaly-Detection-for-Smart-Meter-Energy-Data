
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing


def run_arima(df, order=(1,1,1), title="ARIMA"):
    model = ARIMA(df["y"], order=order)
    model_fit = model.fit()
    pred = model_fit.predict(start=0, end=len(df)-1)
    _plot_forecast(df, pred, title)

def run_sarima(df, order=(1,1,1), seasonal_order=(1,1,1,12), title="SARIMA"):
    model = SARIMAX(df["y"], order=order, seasonal_order=seasonal_order)
    model_fit = model.fit(disp=False)
    pred = model_fit.predict(start=0, end=len(df)-1)
    _plot_forecast(df, pred, title)

def run_holt_winters(df, title="Holt-Winters"):
    model = ExponentialSmoothing(df["y"], seasonal_periods=12, trend="add", seasonal="add")
    model_fit = model.fit()
    pred = model_fit.predict(start=0, end=len(df)-1)
    _plot_forecast(df, pred, title)


def _plot_forecast(df, forecast, title):
    plt.figure(figsize=(14, 5))
    plt.plot(df["ds"], df["y"], label="True")
    plt.plot(df["ds"], forecast, label="Forecast", color="orange")
    plt.title(f"{title} - Forecast")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
