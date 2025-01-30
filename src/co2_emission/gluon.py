from sktime.forecasting.model_selection import temporal_train_test_split
import pandas as pd
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
import os
import matplotlib.pyplot as plt

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


folder_path = '../Data/electric/'
df = read_csv_files_from_folder(folder_path)

# Data Exploration 
df.head()

energy_df = df \
    .rename(columns = {'validto (UTC)' : 'timestamp' ,
                      'emissionfactor (kg CO2/kWh)' : 'target'}) \
    .drop(columns = ['validfrom (UTC)','point', 'granularity', 'percentage','type', 'timezone', 'emission (kg CO2)', 'activity', 'classification', 'capacity (kW)','volume (kWh)']) \
    .dropna() 

energy_df['timestamp'] = pd.to_datetime(energy_df['timestamp'])
energy_df['item_id'] = 1
      
energy_df.head()
energy_df.info()

fig, ax = plt.subplots(figsize=(14, 8))
ax.plot(energy_df['timestamp'], energy_df['target'])
ax.set_xlabel('Date Time')
ax.set_ylabel('Emission Factor')
fig.autofmt_xdate()
plt.tight_layout()
plt.show()

y_train, y_test = temporal_train_test_split(energy_df, test_size=168)
print(len(y_train), len(y_test)) # 34896 168

y_train.head()
y_test.head()
y_test.tail(25)


train_data = TimeSeriesDataFrame.from_data_frame(
    y_train,
    id_column       = "item_id",
    timestamp_column = "timestamp"
)
train_data.head()

predictor = TimeSeriesPredictor(
    prediction_length=144, # 6 days ahead
    path="autogluon",
    target="target",
    eval_metric="MASE",
)

predictor.fit(
    train_data,
    presets="best_quality",
    time_limit=600,
)

predictions = predictor.predict(train_data)
predictions.head()

# Plot time series and the respective forecasts
predictor.plot(y_test, predictions, quantile_levels=[0.1, 0.9], max_history_length=200)
predictor.plot(y_test, predictions, max_history_length=200)

# The test score is computed using the last
# prediction_length=168 timesteps of each time series in test_data
predictor.leaderboard(y_test)

# DeepAR
predictor_deep = TimeSeriesPredictor(target='target', prediction_length=144).fit(
   train_data,
   time_limit=100,
   presets="medium_quality",
   hyperparameters={
      "DeepAR": {},
   },
)

predictions_deep = predictor_deep.predict(train_data)
predictions_deep


predictor_deep.plot(y_test, predictions_deep, quantile_levels=[0.1, 0.9], max_history_length=200)

# plt.figure(figsize=(20, 3))

# item_id = 1
# y_past = train_data.loc[item_id]["target"]
# y_pred = predictions_deep.loc[item_id]
# y_test = y_test[item_id]["target"]

# plt.plot(y_past, label="Past time series values")
# plt.plot(y_pred["mean"], label="Mean forecast")
# plt.plot(y_test, label="Future time series values")

# plt.fill_between(
#     y_pred.index, y_pred["0.1"], y_pred["0.9"], color="red", alpha=0.1, label=f"10%-90% confidence interval"
# )
# plt.legend()