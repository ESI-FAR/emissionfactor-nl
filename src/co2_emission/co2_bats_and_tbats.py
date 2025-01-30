import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from pmdarima import auto_arima
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.performance_metrics.forecasting import *
from sktime.utils.plotting import plot_correlations
from sktime.forecasting.bats import BATS
from sktime.forecasting.tbats import TBATS
from sktime.forecasting.fbprophet import Prophet
from sktime.forecasting.arima import ARIMA

from sktime.forecasting.base import ForecastingHorizon

from sktime.utils.plotting import plot_series

from statsmodels.tsa.seasonal import MSTL
from statsmodels.tsa.seasonal import STL

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

import seaborn as sns

from statsmodels.tsa.stattools import adfuller

from sklearn.metrics import r2_score

from darts import TimeSeries
from darts.utils.statistics import check_seasonality

# sources
#  https://www.kaggle.com/code/somertonman/time-series-classification-using-deep-learning
# https://www.linkedin.com/pulse/tbats-python-tutorial-examples-ikigailabs

def print_metrics(y_true, y_pred, model_name):
    mae_ = mean_absolute_error(y_true, y_pred) # Measures the average magnitude of errors
    rmse_ = mean_squared_error(y_true, y_pred, square_root = True) #  Penalizes large errors more than MAE
    mape_ = mean_absolute_percentage_error(y_true, y_pred) # Measures the percentage difference
    smape_ = mean_absolute_percentage_error(y_true, y_pred, symmetric = True)
    
    dict_ = {'MAE': mae_, 
             'RMSE': rmse_,
             'MAPE': mape_, 
             'SMAPE': smape_ }
    
    df = pd.DataFrame(dict_, index = [model_name])
    return(df.round(decimals = 2))


def read_csv_files_from_folder(folder_path):
    """
    Reads all CSV files from a specified folder and returns a concatenated DataFrame.
    
    Parameters:
    folder_path (str): The path to the folder containing CSV files.
    
    Returns:
    pd.DataFrame: A DataFrame containing the data from all CSV files.
    """
    # List to store individual DataFrames
    dataframes = []

    # Loop over all files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):  
            file_path = os.path.join(folder_path, filename)
            try:
                # Read the CSV into a DataFrame and append it to the list
                df = pd.read_csv(file_path)
                dataframes.append(df)
                print(f"Successfully read {filename} with shape {df.shape}")
            except Exception as e:
                print(f"Error reading {filename}: {e}")
    
    # Concatenate all DataFrames into a single DataFrame
    if dataframes:
        combined_df = pd.concat(dataframes, ignore_index=True)
        print(f"Combined DataFrame shape: {combined_df.shape}")
        return combined_df
    else:
        print("No CSV files were found in the folder.")
        return pd.DataFrame()


pwd = os.getcwd()
print(pwd)

folder_path = '../Data/electric/'
df = read_csv_files_from_folder(folder_path)

# Data Exploration 
df.head()

energy_df = df \
    .rename(columns = {'validto (UTC)' : 'date_time' ,
                      'emissionfactor (kg CO2/kWh)' : 'emissionfactor'}) \
    .drop(columns = ['validfrom (UTC)','point', 'granularity', 'percentage','type', 'timezone', 'emission (kg CO2)', 'activity', 'classification', 'capacity (kW)','volume (kWh)']) \
    .dropna() 

energy_df['date_time'] = pd.to_datetime(energy_df['date_time'])
      
energy_df.head()
energy_df.info()

fig, ax = plt.subplots(figsize=(14, 8))
ax.plot(energy_df['date_time'], energy_df['emissionfactor'])
ax.set_xlabel('Date Time')
ax.set_ylabel('Emission Factor')
fig.autofmt_xdate()
plt.tight_layout()
plt.show()

# Applying multiple seasonal trend decomposition by Loess (MSTL)
mstl = MSTL(energy_df['emissionfactor'], periods=[24, 24*7], iterate = 3)
res = mstl.fit()

res.trend.plot()

resid = res.resid
resid.plot()
plt.rcParams['figure.figsize'] = (10,8)
res.plot(resid = False)
plt.show()

decomposition = STL(energy_df['emissionfactor'], period=24, robust=True).fit()
residuals = decomposition.resid
seasonal_factor = decomposition.seasonal
trend = decomposition.trend
trend.plot()
seasonal_factor.plot()

plot_series(energy_df['emissionfactor'])
plot_series(trend)
plot_series(seasonal_factor)
plot_series(residuals)

energy_df_D = energy_df.copy()

# Step 1: Ensure date_time is in datetime format
energy_df_D['date_time'] = pd.to_datetime(energy_df_D['date_time'])

# Step 2: Set date_time as the index
energy_df_D = energy_df_D.set_index('date_time')

# Step 3: Resample to daily frequency and compute the mean
energy_df_D_resampled = energy_df_D.resample('M').mean()
mstl = MSTL(energy_df_D_resampled['emissionfactor'], periods=[12], iterate = 1)
res = mstl.fit()


# Applying multiple seasonal trend decomposition by Loess (MSTL)
plt.rcParams['figure.figsize'] = (10,8)
res.plot(resid = False)
plt.show()

# Check whether seasonal periods in co2 emission data are significant
series = TimeSeries.from_dataframe(energy_df, time_col='date_time')
is_daily_seasonal,  daily_period  = check_seasonality(series, m=24, max_lag=400, alpha=0.01)
is_weekly_seasonal, weekly_period = check_seasonality(series, m=168, max_lag=400, alpha=0.01)
is_yearly_seasonal, yearly_period = check_seasonality(series, m=8760, max_lag=10000, alpha=0.01)

print(f'Daily seasonality: {is_daily_seasonal} - period = {daily_period}')
print(f'Weekly seasonality: {is_weekly_seasonal} - period = {weekly_period}')
print(f'Yearly seasonality: {is_yearly_seasonal} - period = {yearly_period}')

# modeling ----
y = energy_df['emissionfactor']

# forecasting horizon 7 x 24 hours
fh = np.arange(1, 169) 

# Autocorrelations
fig,ax = plot_correlations(y, lags = 170)
# r24 is higher than the other lags due to the daily pattern in the data. The data also have a trend because the autocorrelations for small lags tend to be large and positive because observations nearby in time are also nearby in value.

# Selection criteria for the order of ARIMA model:
# p : Lag value where the Partial Autocorrelation (PACF) graph cuts off or drops to 0 for the 1st instance.
# d : Number of times differencing is carried out to make the time series stationary.
# q : Lag value where the Autocorrelation (ACF) graph crosses the upper confidence interval for the 1st instance.
p = 2, d = 0, q = 2

# Augmented Dickey-Fuller unit root test
# The null hypothesis (H₀) of the ADF test is that the series has a unit root (i.e., it is non-stationary).
# The alternative hypothesis (H₁) is that the series is stationary.
# To reject the null hypothesis, you have two criteria:
# - p-value: If p-value < 0.05, reject H₀ (time series is stationary).
# - ADF Statistic: If ADF Statistic < Critical Value, reject H₀ (time series is stationary).
adfuller(y)

# Test Statistic = -13.66
# Critical Value at 1% = -3.4305
# Since -13.66 < -3.4305, the null hypothesis at the 1% significance level is rejected, 
# meaning the series is stationary.

# The second value, 2.87e-17, is the p-value for the test.
# The p-value tells you the probability of observing the given test statistic under the null
# hypothesis. If the p-value is less than 0.05, you reject the null hypothesis, meaning
# the series is stationary. Here, the p-value is extremely small (2.87e-17), meaning the 
# test provides very strong evidence to reject the null hypothesis of non-stationarity.
# This means the series is stationary at the 1%, 5%, and 10% confidence levels, so no 
# differencing is required.


# test train split ----
y_train, y_test = temporal_train_test_split(y, test_size=168)
print(len(y_train), len(y_test)) # 34896 168

# Baseline prediction
y_pred_baseline = y_train[-168:].values

# Reduce the training set to last 10 weeks
#y_train = y_train[-1680:]
y_train = y_train[-504:] # last 3 weeks

y_train.plot()
y_test.plot()

print(f"Variance of y_train: {y_train.var():.4f}")
print(f"Variance of y_test: {y_test.var():.4f}")

print(len(y_train))
print(len(y_test))
print(y_train.head())

# BATS ---- (this takes a long time!! Skip this and load the zipped model below)
forecaster_bats = BATS(use_box_cox=False,
                       use_trend=True,
                       use_damped_trend=False,
                       sp=[24, 168],
                       use_arma_errors=True, 
                       n_jobs=-1)
bats_model = forecaster_bats.fit(y_train)

# Saving fitted BATS model
bats_model.save(path="models/bats_model2")

# Loading fitted BATS model
bats_model = forecaster_bats.load_from_path("models/bats_model2.zip")
bats_model.is_fitted # True

# forecasting horizon for the training period
fh_train = ForecastingHorizon(y_train.index, is_relative=False)
len(fh_train) # 1680

# Get predictions for the training data
fitted_values_bats = bats_model.predict(fh = fh_train)

# Calculate residuals
residuals_train = y_train - fitted_values_bats

# Plot the residuals over time
plt.figure(figsize=(12, 6))
plt.plot(y_train.index, residuals_train, label='BATS Residuals training', color='red')
plt.axhline(0, color='black', linestyle='dashed')
plt.title('BATS Residuals Over Time training')
plt.show()

#  Autocorrelation plot for residuals
plot_acf(residuals_train, lags=50)
plt.title('ACF of BATS Residuals')
plt.show()

# calculate R²
r2_bats_train = r2_score(y_train, fitted_values_bats)
print(f'R² for BATS on training data: {r2_bats_train:.4f}') # R² training data: 0.9807

# Get the actual values and the BATS predictions on the test set
y_pred_BATS = bats_model.predict(fh) # forecasting horizon 7 x 24 hours

# Calculate residuals
residuals_test = y_test - y_pred_BATS

# Plot residuals over time
plt.figure(figsize=(10, 5))
plt.plot(y_test.index, residuals, label="Residuals", color="purple")
plt.axhline(0, color='black', linestyle='--', label="Zero Error")
plt.xlabel("Time")
plt.ylabel("Residuals")
plt.title("Residual Plot: Test Data")
plt.legend()
plt.show()

residual_variance = residuals.var()
test_variance = y_test.var()
print(f"Residual Variance: {residual_variance:.4f}")
print(f"Test Variance: {test_variance:.4f}")

# Calculate MAE and RMSE
mae = mean_absolute_error(y_test, y_pred_BATS)
rmse = mean_squared_error(y_test, y_pred_BATS, squared=False)

print(f"MAE (Mean Absolute Error): {mae:.4f}")
print(f"RMSE (Root Mean Squared Error): {rmse:.4f}")

r2_bats_test = r2_score(y_test, y_pred_BATS)
print(f'R² for BATS on test data: {r2_bats_test:.4f}')
# R² for BATS on test data: -0.0672


# TBATS  ----
# we specify the seasonal periods, which are 24 (for the daily seasonality) and 168 
# (for the weekly seasonality) [and 8760 (for the yearly seasonality)]
forecaster_tbats = TBATS(use_box_cox=False,
                 use_trend=True,
                 use_damped_trend=True,
                 sp=[24, 168],
                 use_arma_errors = False,
                 n_jobs=10)   

# Fit to training data
tbats_model = forecaster_tbats.fit(y_train)
tbats_model.is_fitted # True

fitted_params = tbats_model.get_fitted_params(deep=True)
print(fitted_params)

# save model
tbats_model.save(path="models/tbats_model2")

# Loading the fitted TBATS model
tbats_model = forecaster_tbats.load_from_path("models/tbats_model2.zip")

# make predictìons stored in y_pred_TBATS
y_pred_TBATS = tbats_model.predict(fh) # forecasting horizon 7 x 24 hours

# get the residuals for the test set 
res = forecaster_tbats.predict_residuals(y_test)
res.plot()

print(f'Length of y_true: {len(y_test)}')
print(f'Length of y_pred: {len(y_pred_TBATS)}')
print_metrics(y_test, y_pred_TBATS, 'TBATS Forecaster')

# ARIMA/SARIMAX ----
# find the best model's parameters
auto_model = auto_arima(
    y_train, 
    seasonal=True, 
    m=24,  # Use daily seasonality as the baseline
    stepwise=True, 
    trace=True, 
    error_action='ignore', 
    suppress_warnings=True
)

print(auto_model.summary())

# Extract residuals from the model
residuals = auto_model.resid()

# Plot residuals
plt.figure(figsize=(12, 6))
plt.subplot(211)
plt.plot(residuals)
plt.title('Residuals')
plt.subplot(212)
sns.histplot(residuals, kde=True)
plt.title('Residual Distribution')
plt.show()

# Plot ACF and PACF of residuals
plot_acf(residuals)
plt.show()

plot_pacf(residuals)
plt.show()

# running the best model derived from the auto_arima results
forecaster_arima = ARIMA(order=(1, 1, 1), seasonal_order = (1, 0, 1, 24), suppress_warnings=True)
forecaster_arima.fit(y_train)
forecaster_arima.summary()
forecaster_arima.save(path="models/sarimax_model")
y_pred_SARIMAX = forecaster_arima.predict(fh)

# plot and compare the predictions
fig, ax = plt.subplots(figsize=(14, 8))
ax.plot(y_train, ls='-', label='Train')
ax.plot(y_test, ls='-', label='Test')
ax.plot(y_test.index, y_pred_baseline, ls=':', label='Baseline')
ax.plot(y_pred_BATS,  ls='--', label='BATS')
ax.plot(y_pred_TBATS, ls='-.', label='TBATS')
ax.plot(y_pred_SARIMAX, ls='--', label='SARIMAX') 

ax.set_xlabel('date time')
ax.set_ylabel('Co2 emission')
ax.legend(loc='best')

fig.autofmt_xdate()
plt.tight_layout()
plt.xlim(34100, 34400)
plt.ylim(0, 0.45)
plt.show()

# comparing all models based on MAPE
def mape(y_true, y_pred):
    return round(np.mean(np.abs((y_true - y_pred) / y_true)) * 100,2)

mape_baseline = mape(y_test, y_pred_baseline)
mape_BATS     = mape(y_test, y_pred_BATS)
mape_TBATS    = mape(y_test, y_pred_TBATS)
mape_SARIMAX  = mape(y_test, y_pred_SARIMAX)

print(f'MAPE from baseline: {mape_baseline}')
print(f'MAPE from BATS: {mape_BATS}')
print(f'MAPE from TBATS: {mape_TBATS}')
print(f'MAPE from SARIMAX: {mape_SARIMAX}')

fig, ax = plt.subplots()
x = ['Baseline','BATS','TBATS','SARIMAX']
y = [mape_baseline, mape_BATS, mape_TBATS, mape_SARIMAX]

ax.bar(x, y, width=0.4)
ax.set_xlabel('Models')
ax.set_ylabel('MAPE (%)')
ax.set_ylim(0, 50)

for index, value in enumerate(y):
    plt.text(x=index, y=value + 1, s=str(round(value,2)), ha='center')
plt.tight_layout()
