# LSTM multistep univariate forecasting
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error

from tensorflow.keras.models import load_model
from tensorflow.keras.layers import LSTM

from keras.layers import LSTM,Flatten,Dense,Dropout,SpatialDropout1D,Bidirectional,LeakyReLU
from keras.layers import Input,Conv1D,MaxPool1D
from keras import regularizers
from keras.callbacks import ModelCheckpoint,EarlyStopping
from keras.models import Sequential
import tensorflow as tf
from tensorflow import keras

import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf

import pickle as pkl

import plotly.express as px

import numpy as np
from datetime import datetime, timedelta

import plotly as py
import plotly.offline as pyoff
import plotly.graph_objs as go

from sklearn.preprocessing import MinMaxScaler

os.getcwd()

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
    
    if dataframes:
        combined_df = pd.concat(dataframes, ignore_index=True)
        print(f"Combined DataFrame shape: {combined_df.shape}")
        return combined_df
    else:
        print("No CSV files were found in the folder.")
        return pd.DataFrame()

folder_path = '../Data/electric/'
df_energy = read_csv_files_from_folder(folder_path)

# Data Exploration 
df_energy.head()
df_energy.tail()

# Data Cleaning
df_energy = df_energy \
    .rename(columns = {'validto (UTC)' : 'date_time' ,
                      'emissionfactor (kg CO2/kWh)' : 'RRP'}) \
    .drop(columns = ['validfrom (UTC)','point', 'granularity', 'percentage','type', 'timezone', 'emission (kg CO2)', 'activity', 'classification', 'capacity (kW)','volume (kWh)']) \
    .dropna() 

#df_energy['date_time'] = pd.to_datetime(df_energy['date_time'])
df_energy['time']=pd.to_datetime(df_energy['date_time'],errors='coerce')
df_energy['day']=df_energy['time'].dt.day
df_energy.drop(columns=['date_time'],inplace=True)
df_energy.set_index(df_energy['time'],inplace=True)
df_energy.head()

# Displaying the number of unique values in each column of the DataFrame.
df_energy.nunique()

# Interactive exploration
fig= px.line(df_energy, x='time', y='RRP', title='CO2 emission with slider')
fig.update_xaxes(rangeslider_visible=True)
fig.show()

df_energy.tail(1354)

df_energy=df_energy[df_energy['time'] >= '2023-01-01 14:00:00']
df_energy.shape

# Split the data into three subsets
n = len(df_energy)
train_df = df_energy[0:int(n*0.7)].copy()
val_df   = df_energy[int(n*0.7):int(n*0.9)].copy()
test_df  = df_energy[int(n*0.9):].copy()

train_df
val_df
test_df

# Ensure train_df, val_df, and test_df contain sufficient data points to cover n_in + n_out!
len(train_df), len(val_df), len(test_df)

# Fit the scaler on the training data
scaler = MinMaxScaler(feature_range=(0, 1))
train_df['RRP'] = scaler.fit_transform(train_df[['RRP']])

# Scale validation and test data using the already-fitted scaler
val_df['RRP'] = scaler.transform(val_df[['RRP']])
test_df['RRP'] = scaler.transform(test_df[['RRP']])

# Exploratory Data Analysis ----

# Autocorrelation (ACF) plot
fig, ax = plt.subplots(figsize=(6, 3))
plot_acf(df_energy['RRP'], lags=20, ax=ax)
plt.title('ACF Plot')
plt.show()

# Plots the 'RRP' column from the test_df DataFrame.
test_df['RRP'].plot()

# Displaying the first few rows of the training dataset 'train_df'
train_df.head()

# An interactive line plot of 'RRP' against time for the training dataset, allowing slider interaction for exploration.
fig= px.line(train_df, x = 'time', y = 'RRP', title = 'CO2 with slider')
fig.update_xaxes(rangeslider_visible=True)
fig.show()

# An interactive line plot of 'RRP' against time for the training dataset, allowing slider interaction for exploration.
fig= px.line(test_df, x = 'time', y = 'RRP', title = 'CO2 with slider')
fig.update_xaxes(rangeslider_visible=True)
fig.show()


# A line plot of 'RRP' against time for the training dataset with a range 
# selector allowing users to select different time ranges (from 1 to 7 days
# or all data),
fig= px.line(train_df,x='time',y='RRP',title='RRP with slider')
lt=[dict(count=i,label=str(i)+'D',step="day",stepmode='todate') for i in range(1,8)]
lt.append(dict(step='all'))
fig.update_xaxes(rangeslider_visible=True,
                 rangeselector=dict(
                     buttons=lt
                 ))
fig.show()

# A lag plot for the 'RRP' column in the training dataset with a lag of 1, using the lag_plot function.
pd.plotting.lag_plot(train_df['RRP'],lag=1)

# A lag plot for the 'RRP' column in the training dataset with a lag of 24 time periods (presumably hours).
pd.plotting.lag_plot(train_df['RRP'],lag=24)

# A lag plot for the 'RRP' column in the training dataset with a lag of 12, using the lag_plot function.
pd.plotting.lag_plot(train_df['RRP'],lag=12)

# A lag plot for the 'RRP' column in the training dataset with a lag of 48, using the lag_plot function.
pd.plotting.lag_plot(train_df['RRP'],lag=48)

# A lag plot for the 'RRP' column in the training dataset with a lag of 48*7 time periods (presumably 7 days in hours).
pd.plotting.lag_plot(train_df['RRP'],lag=48*7)

# An autocorrelation plot for the 'RRP' column in the training dataset
pd.plotting.autocorrelation_plot(train_df['RRP'])

# An autocorrelation plot for the 'RRP' column in the training dataset after resampling it to a daily frequency and taking the mean for each day.
pd.plotting.autocorrelation_plot(train_df['RRP'].resample('1d').mean())


# Data preprocessing and train-test split ----

# The "moving_average" function calculates the moving average of a given dataset data using a specified window size window.
# A set of weights are created for the moving average and then applies the convolution operation to compute the moving average.
# Applying moving average only to training data can act as a form of regularization.
# It helps reduce noise in the training data while keeping validation/test data in their raw form.
# This approach forces the model to learn more robust patterns rather than potentially overfitting to noise.

def moving_average(data, window):
    weights = np.repeat(1.0, window) / window
    return np.convolve(data, weights, mode='same')

# Applying the moving average function to the 'RRP' column in the training dataset with a window size of 5, smoothing out the data, and assigns the result back to the 'RRP' column in the DataFrame.

train_df['RRP']=moving_average(train_df['RRP'].values, window=5)
train_df

# This function converts a time series column into a supervised learning 
# format, generating input and output sequences based on specified parameters.
# transform list into supervised learning format
def series_to_supervised(data, n_in=168, n_out=24): # change to 168
    '''   
    Handles sequences where the target is a multi-step forecast. 
    data: Time series column
    n_in: number of input time steps
    n_out: number of output time steps (forecast horizon)
    '''
    df = pd.DataFrame(data)
    cols = list()
    
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        
    # put it all together
    agg = pd.concat(cols, axis=1)
    
    # drop rows with NaN values
    agg.dropna(inplace=True)
    
    return agg.values

# Stacked LSTM-Conv Adam Model ----

# Creating a neural network model with Bidirectional LSTM and Convolutional
# layers for time series forecasting

# We're only using the CO2 emission ('RRP') as input
num_features = 1

# Model architecture for multi-step forecasting
def create_multistep_lstm_model(window_length, forecast_horizon):
    model = Sequential([
        Input(shape=(window_length, 1)),
        # BLSTM-Conv Block 1
        Bidirectional(LSTM(128, return_sequences=True)),
        Conv1D(128, kernel_size=3, activation='relu', padding='same'),
        MaxPool1D(2),
        # BLSTM-Conv Block 2
        Bidirectional(LSTM(64, return_sequences=True)),
        Conv1D(64, kernel_size=3, activation='relu', padding='same'),
        MaxPool1D(2),
        # BLSTM-Conv Block 3
        Bidirectional(LSTM(32, return_sequences=True)),
        Conv1D(32, kernel_size=3, activation='relu', padding='same'),
        MaxPool1D(2),
        Dropout(0.05),
        Flatten(),
        Dense(forecast_horizon, activation='relu')  # Output layer matches forecast horizon
    ])
    
    model.compile(loss='mae', # MeanAbsoluteError
                 optimizer='adam',
                 metrics=['mse'])
    return model

# Data preparation
window_length = 168 # represents 168 hours of historical data with which to predict the future
forecast_horizon = 24 # this means we want to predict 24 hours in the future

# Displaying the first 5 values of the 'RRP' column in the training dataset
train_df.RRP[:5]

# Prepare time series data for modeling by converting it into a supervised learning format with a 
# specified window length (168) and forecast horizon (24)
train_data = series_to_supervised(train_df.RRP, n_in=window_length, n_out=forecast_horizon)
val_data   = series_to_supervised(val_df.RRP,   n_in=window_length, n_out=forecast_horizon)
test_data  = series_to_supervised(test_df.RRP,  n_in=window_length, n_out=forecast_horizon)

len(train_data), len(val_data), len(test_data)

# The output of series_to_supervised has rows where:
# - The first window_length columns (168 in this case) are the input features.
# - The next forecast_horizon columns (24 in this case) are the target values to predict.

# train_data: [x1, x2, ..., x168 | y1, y2, ..., y24]
#             ^ Features ^       ^ Targets ^

# Split features X and targets y
# X_train: Contains the sequences of window_length time steps used as input for the model.
# These represent historical data points that the model will use to make predictions.
X_train = train_data[:, :window_length] # selects the first 168 columns of each row (input features)

# y_train: Contains the forecast_horizon (24) target values for each sequence in X_train.
# These represent the actual future values the model is tasked with predicting.
y_train = train_data[:, window_length:] # selects the last 24 columns of each row (target values)


# Example: Let's assume window_length = 3 and forecast_horizon = 2 for simplicity

# Input Time Series:
# data = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

# After series_to_supervised:
# train_data =
# [
#  [0.1, 0.2, 0.3 | 0.4, 0.5],  # Row 1
#  [0.2, 0.3, 0.4 | 0.5, 0.6],  # Row 2
#  [0.3, 0.4, 0.5 | 0.6, 0.7],  # Row 3
#  [0.4, 0.5, 0.6 | 0.7, 0.8]   # Row 4
# ]

# Splitting into X_train and y_train:
# X_train = [
#  [0.1, 0.2, 0.3],  # Features for Row 1
#  [0.2, 0.3, 0.4],  # Features for Row 2
#  [0.3, 0.4, 0.5],  # Features for Row 3
#  [0.4, 0.5, 0.6]   # Features for Row 4
# ]

# y_train = [
#  [0.4, 0.5],  # Targets for Row 1
#  [0.5, 0.6],  # Targets for Row 2
#  [0.6, 0.7],  # Targets for Row 3
#  [0.7, 0.8]   # Targets for Row 4
# ]

# Similarly, split the validation and test datasets into X and y
X_val, y_val     = val_data[:, :window_length], val_data[:, window_length:]
X_test, y_test   = test_data[:, :window_length], test_data[:, window_length:]
len(X_train), len(X_val), len(X_test)

# Reshape input data for LSTM [samples, timesteps, features]
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
len(X_train), len(X_val), len(X_test)

# Create and train model
model = create_multistep_lstm_model(window_length, forecast_horizon)
 
# Defining the directory path for storing model files and results
temp_folder = "C:/Projects/TeSoPs/Syntax/Model Files/co2_emission/"

# Defining the directory path for storing pickle files
temp = 'C:/Projects/TeSoPs/Syntax/Model Files/Temp'

# Defining a checkpoint to save the best model during the training at the 
# directory where all models are saved
checkpoint = ModelCheckpoint(temp_folder + '/lstm_conv_adam_multistep.keras',
                           monitor='val_mse', 
                           save_best_only=True, 
                           verbose=1)

# Training the model
history = model.fit(X_train, y_train,
                   batch_size=512,
                   validation_data=(X_val, y_val),
                   epochs=200,
                   callbacks=[checkpoint])

# Saving the training history of the Stacked LSTM-Conv Adam model
pkl.dump(history.history, open(temp + '/results_multistep.pkl', 'wb'))

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch

plot_data = [
    go.Scatter(
        x=hist['epoch'],
        y=hist['loss'],
        name='loss'
    )
    
]

plot_layout = go.Layout(
        title='Training loss'
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
pyoff.iplot(fig)

pred_list = []
n_input = 168
n_features = 1

batch = train[-n_input:].reshape((1, n_input, n_features))

for i in range(n_input):   
    pred_list.append(model.predict(batch)[0]) 
    batch = np.append(batch[:,1:,:],[[pred_list[i]]],axis=1)


# Evaluation ----
# Loading the model file from the specified location with custom loss function
model = load_model(temp_folder + '/lstm_conv_adam_multistep.keras')
model.summary()

# Making predictions on the training and testing datasets using the loaded model
train_pred = model.predict(X_train) # yhat train
test_pred = model.predict(X_test)   # yhat test 

# test for similar shapes
y_test.shape, test_pred.shape

# shape of the prediction arrays
train_pred.shape, test_pred.shape

# Inverse-transform training and test predictions to the original scale (yhat, i.e. predicted co2)
train_pred_original_scale = scaler.inverse_transform(train_pred)
test_pred_original_scale = scaler.inverse_transform(test_pred)

# Inverse-transform training and test true values (y, i.e. true co2)
y_train_original_scale = scaler.inverse_transform(y_train)
y_test_original_scale = scaler.inverse_transform(y_test)

# Plotting the predicted vs actual values for a single forecast horizon
plt.figure(figsize=(12, 6))

# Choose a sample index for comparison
sample_index = 1562  # You can change this to visualize different examples
plt.plot(range(forecast_horizon), y_test_original_scale[sample_index], label='Actual Values', marker='o')
plt.plot(range(forecast_horizon), test_pred_original_scale[sample_index], label='Predicted Values', marker='x')

plt.title(f'Forecast vs Actual for Sample {sample_index}')
plt.xlabel('Forecast Horizon (timesteps)')
plt.ylabel('RRP (Original Scale)')
plt.legend()
plt.grid(True)
plt.show()

# or 

# Visualize multiple samples
for i in range(3):  # Plot the first 3 samples
    plt.figure()
    plt.plot(y_test[i], label=f'Actual {i}')
    plt.plot(test_pred[i], label=f'Predicted {i}')
    plt.legend()
    plt.show()
    
# or

# ---------------------------------------------------------------------
# plot that:
# - Combines the training and test datasets on the same timeline
# - Starts the forecast from the last point of the training dataset
# - Overlays the multi-step predictions on the test data for comparison

# Define how many training points to include before the forecast
num_train_points_to_show = 50  # Adjust this for your desired context window

# Extract the section of the training data to display
train_section = train_df.RRP[-num_train_points_to_show:]

# Predict multi-step forecast starting from the last training point
last_train_window = train_df.RRP[-window_length:].values.reshape((1, window_length, 1))
forecast = model.predict(last_train_window).flatten()  # Flatten for easier plotting

# Extract the corresponding section of the test data
test_section = test_df.RRP[:forecast_horizon]

# Plot the relevant section
plt.figure(figsize=(12, 6))

# Plot the selected portion of training data
plt.plot(range(-num_train_points_to_show, 0), train_section, label="Training Data", color="blue")

# Plot the test data
plt.plot(range(0, len(test_section)), test_section, label="Test Data", color="orange")

# Plot the forecast
plt.plot(range(0, len(forecast)), forecast, label="Forecast", color="green", linestyle="--")

# Add labels, legend, and grid
plt.xlabel("Time")
plt.ylabel("Values")
plt.title("Zoomed-in View of Forecasts and Test Data")
plt.legend()
plt.grid()
plt.show()

# Calculate Mean Absolute Error for Each Step in Forecast Horizon
mae_per_step = np.mean(np.abs(y_test_original_scale - test_pred_original_scale), axis=0)

# Plot MAE per forecast horizon step
plt.figure(figsize=(12, 6))
plt.plot(range(forecast_horizon), mae_per_step, marker='o')
plt.title('Mean Absolute Error per Forecast Horizon Step')
plt.xlabel('Forecast Horizon (timesteps)')
plt.ylabel('Mean Absolute Error')
plt.grid(True)
plt.show()

# Calculating Mean Squared Error (MSE) and Mean Absolute Error (MAE) for the testing dataset
mse = mean_squared_error(y_test,test_pred)
mae = mean_absolute_error(y_test,test_pred)

test_mae = mean_absolute_error(y_test_original_scale, test_pred_original_scale)
test_mse = mean_squared_error(y_test_original_scale, test_pred_original_scale)
test_mse, test_mae

# Calculating the prediction errors
error = np.array([y_test_original_scale[i]-test_pred_original_scale[i] for i in range(len(y_test_original_scale))]).flatten()
error

# Note: Saving test_pred and error into pickle file to use them for visualization
# in result visualisation section in Error_Correction_and_EAdam notebook. 
# This is done intentionally to remove dependency of running the entire notebook
# for result visualisation.
pkl.dump(test_pred, open(temp + '/test_pred.pkl','wb'))
pkl.dump(error, open(temp + '/error.pkl','wb'))

# Displaying the first 5 actual and predicted values
y_test[:5], test_pred[:5]

# Plotting the prediction errors
plt.plot(error)
plt.show()

# Plotting the predicted and actual values for the testing dataset
plt.figure(figsize=(16, 4))
plt.plot(test_pred_original_scale,label='Predicted y value')
plt.plot(y_test_original_scale,label='Actual y value')
plt.title('Forecast vs Actual')
plt.legend()
plt.rcParams["figure.figsize"] = (15,3)
plt.show()

# Calculating and displaying the Standard Deviation of Errors (SDE)
def sde(y_true,y_pred):
  mean_diff=(y_true-y_pred).mean()
  error=(y_true-y_pred-mean_diff)**2
  return np.sqrt(error.mean())

test_predSDE = np.concatenate( test_pred, axis=0 )
sde(y_test,test_predSDE)

# Calculating and displaying the Symmetric Mean Absolute Percentage Error (SMAPE).
def smape(A, F):
    return 100/len(A) * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F)))
smape(y_test,test_predSDE)

# Calculating and displaying the Mean Absolute Percentage Error (MAPE)
def mape(actual, forecast):
    return np.mean(np.abs((actual - forecast) / actual)) * 100.0
mape(y_test,test_predSDE)

# Table showing results obtained for Stacked LSTM-Conv Adam Strategy:
# Input metrics
metrics = ["MSE", "MAE", "SDE", "SMAPE", "MAPE"]
values = [mse,mae, sde(y_test,test_predSDE),smape(y_test,test_predSDE),mape(y_test,test_predSDE)]

# Create a DataFrame with metrics as columns
df = pd.DataFrame([values], columns=metrics)

# Add a custom row label (e.g., "Adam")
df.insert(0, "Metrics", ["Adam"])

# Format the numbers for readability (rounding)
df[metrics] = df[metrics].map(lambda x: f"{x:.3f}")

# Print the table
print(df.to_string(index=False))


# Model retraining and forecasting into the unknown future ----

# For forecasting beyond the test period, we would typically want to retrain 
# the model on all available data (training + validation + test) because:
# - More recent data often contains the most relevant patterns for future prediction
# - we want to use all available information when making real forecasts
# - The test set was only held out to evaluate model performance - once we've 
#   validated the model's effectiveness, we can use that data for training

# Prepare all data for training
full_data = series_to_supervised(df_energy.RRP, n_in = window_length)

X_full, y_full = full_data[:, :window_length], full_data[:, window_length:]
#X_full_data = full_data.reshape((full_data.shape[0], full_data.shape[1], 1))

X_full.shape, y_full.shape

# Create and train model
full_model = create_multistep_lstm_model(window_length, forecast_horizon)


# Defining a checkpoint to save the model during training
checkpoint = ModelCheckpoint(
    temp_folder + '/lstm_conv_adam_full_model_alldata.keras',
    monitor='loss',  # Using loss instead of val_mse since we don't have validation data
    save_best_only=True,
    verbose=1
)

# Training the model on full dataset
full_model_history = full_model.fit(
    X_full, 
    y_full, 
    batch_size=512, 
    epochs=200, 
    callbacks=[checkpoint]
)

hist = pd.DataFrame(full_model_history.history)
hist['epoch'] = full_model_history.epoch

plot_data = [
    go.Scatter(
        x=hist['epoch'],
        y=hist['loss'],
        name='loss'
    )
    
]

plot_layout = go.Layout(
        title='Training loss'
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
pyoff.iplot(fig)

# Save the training history
pkl.dump(full_model_history.history, open(temp + '/adam_results_full_model.pkl', 'wb'))

# training dataset
#full_model = load_model(temp_folder + '/lstm_conv_adam_model_2_TV.keras')

# full dataset
full_model = load_model(temp_folder + '/lstm_conv_adam_full_model_alldata.keras')
full_model.summary()

full_model = model

def forecast_future_week(model, last_sequence, start_date, hours = 168):
    """
    Forecasts values for a week beyond the last available data point.
    
    Parameters:
    model: trained keras model
    last_sequence: last window_length values from the test set
    start_date: datetime of the first forecast point
    hours: number of hours to forecast (default=168 for one week)
    
    Returns:
    tuple: (forecasted values, datetime index)
    """
    # Initialize arrays to store predictions and dates
    predictions = []
    dates = []
    
    # Create copy of the last sequence to avoid modifying original
    current_sequence = last_sequence.copy()
    
    # Generate predictions for each hour
    for i in range(hours):
        # Reshape the sequence for model input (samples, timesteps, features)
        model_input = current_sequence.reshape(1, len(current_sequence), 1)
        
        # Make prediction
        next_pred = model.predict(model_input, verbose=0)[0][0]
        
        # Store prediction and corresponding date
        predictions.append(next_pred)
        dates.append(start_date + timedelta(hours=i))
        
        # Update sequence for next prediction (remove oldest, add new prediction)
        current_sequence = np.roll(current_sequence, -1)
        current_sequence[-1] = next_pred
    
    return np.array(predictions), dates

# Get the last window_length values from the entire dataset
last_sequence = X_full[-1]

# Use this retrained model for forecasting
forecast_start = df_energy.index[-1] + timedelta(hours=1)
future_predictions, future_dates = forecast_future_week(full_model, last_sequence, forecast_start)

# Create dataframe with results
future_df = pd.DataFrame({
    'time': future_dates,
    'forecast': future_predictions
})
future_df.set_index('time', inplace=True)

# Plot the results: last weeks of actual data and the forecast
plt.figure(figsize=(15, 6))
# Plot last weeks of actual data
last_week_mask = df_energy.index >= (forecast_start - timedelta(days=28))
plt.plot(df_energy[last_week_mask].index, df_energy[last_week_mask]['RRP'], 
         label='Actual', color='blue')
# Plot forecasted values
plt.plot(future_df.index, future_df['forecast'], 
         label='Forecast', color='red', linestyle='--')
plt.axvline(x=forecast_start, color='gray', linestyle=':', label='Forecast Start')
plt.title('Last Week of Actual Data vs One Week Forecast')
plt.legend()
plt.grid(True)
plt.show()

# Display the forecasted values
print("\nForecasted values for the next week:")
print(future_df)

# Start with the raw data
complete_df = df_energy.copy()

# Ensure data is properly sorted by time
complete_df = complete_df.sort_index()

# Data preparation parameters
window_length = 24  # Adjust if needed
forecast_horizon = 168
num_features = 1

# Scale the raw data (instead of using moving average)
scaler = MinMaxScaler()
complete_df['RRP'] = scaler.fit_transform(complete_df['RRP'].values.reshape(-1, 1))

# Prepare complete dataset
complete_data = series_to_supervised(complete_df.RRP, 
                                         n_in=window_length, 
                                         n_out=forecast_horizon)

# Split features and targets
X_complete = complete_data[:, :window_length]
y_complete = complete_data[:, window_length:]

# Reshape input data for LSTM [samples, timesteps, features]
X_complete = X_complete.reshape((X_complete.shape[0], X_complete.shape[1], num_features))

# Create fresh model
final_model = create_multistep_lstm_model(window_length, forecast_horizon)

# Training optimization settings
batch_size = 512
epochs = 200

# Early stopping to prevent overfitting
early_stopping = EarlyStopping(
    monitor='loss',
    patience=20,
    restore_best_weights=True
)

# Train final model
final_history = final_model.fit(
    X_complete, 
    y_complete,
    batch_size=batch_size,
    epochs=epochs,
    callbacks=[early_stopping],
    verbose=1
)

# Save the final model
final_model.save(temp_folder + '/lstm_conv_adam_multistep_final.keras')

# Function to make future predictions
# To make predictions, remember to inverse transform the scaled values
def predict_future_sequence(model, last_known_values, scaler, num_features=1):
    """
    Make future predictions using the last known values
    
    Parameters:
    model: trained model
    last_known_values: array of the last window_length values (raw values)
    scaler: fitted MinMaxScaler
    num_features: number of input features
    
    Returns:
    Array of predictions for the next forecast_horizon steps (in original scale)
    """
    # Scale the input
    scaled_input = scaler.transform(last_known_values.reshape(-1, 1))
    
    # Reshape input for prediction
    input_data = scaled_input[-window_length:].reshape(1, window_length, num_features)
    
    # Make prediction
    scaled_prediction = model.predict(input_data)
    
    # Inverse transform the prediction
    future_prediction = scaler.inverse_transform(scaled_prediction.reshape(-1, 1))
    
    return future_prediction.flatten()

# Make future predictions
last_known = complete_df['RRP'].values[-window_length:]
future_sequence = predict_future_sequence(final_model, last_known, scaler)

# Create future dates for plotting
last_date = complete_df.index[-1]
future_dates = pd.date_range(start=last_date + pd.Timedelta(hours=1), 
                           periods=forecast_horizon, 
                           freq='H')

# Plot the results
plt.figure(figsize=(15, 6))
plt.plot(complete_df.index[-168:], complete_df['RRP'].values[-168:], 
         label='Historical Data', color='blue')
plt.plot(future_dates, future_sequence, 
         label='Forecast', color='red', linestyle='--')
plt.title('CO2 Emission Forecast')
plt.xlabel('Date')
plt.ylabel('Emission Factor')
plt.legend()
plt.grid(True)
plt.show()