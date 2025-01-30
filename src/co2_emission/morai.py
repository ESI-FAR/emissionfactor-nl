# Do imports
from sktime.forecasting.moirai_forecaster import MOIRAIForecaster
from sktime.datasets import load_forecastingdata
from sktime.utils.plotting import plot_series

from sktime.forecasting.model_selection import temporal_train_test_split

import matplotlib.pyplot as plt

import pandas as pd
import os

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
    .rename(columns = {'validto (UTC)' : 'date_time' ,
                      'emissionfactor (kg CO2/kWh)' : 'emissionfactor'}) \
    .drop(columns = ['validfrom (UTC)','point', 'granularity', 'percentage','type', 'timezone', 'emission (kg CO2)', 'activity', 'classification', 'capacity (kW)','volume (kWh)']) \
    .dropna() 

energy_df['date_time'] = pd.to_datetime(energy_df['date_time'])
      
energy_df.head()
energy_df.info()

energy_df = energy_df \
    .set_index('date_time')

y = energy_df

# test train split ----
y_train, y_test = temporal_train_test_split(y, test_size=168)

# Check and infer the frequency
if y_train.index.inferred_freq is None:
    print("Frequency is missing, setting it explicitly.")
    y_train.index = pd.date_range(start=y_train.index[0], periods=len(y_train), freq="H")  # Adjust "H" if needed


# Load data
# data = load_forecastingdata("australian_electricity_demand_dataset")[0]
# data = data.set_index("series_name")
# series = data.loc["T1"]["series_value"]
# y = pd.DataFrame(series, index=pd.date_range("2006-01-01", periods=len(series), freq="30min"))
# y = y.resample("H").mean()
# series2 = data.loc["T2"]["series_value"]
# y2 = pd.DataFrame(series2, index=pd.date_range("2006-01-01", periods=len(series), freq="30min"))
# y2 = y2.resample("H").mean()

# Initialise the forecasters
morai_forecaster = MOIRAIForecaster(
    checkpoint_path=f"Salesforce/moirai-1.0-R-small"
)

# Fit the model (in most cases no fitting happens, it is about API consistency)
morai_forecaster.fit(y=y_train, fh=range(1, 168))

pred_moirai = morai_forecaster.predict(y=y_test[:-168])

# plot and compare the predictions
fig, ax = plt.subplots(figsize=(14, 8))
ax.plot(y_train, ls='-', label='Train')
ax.plot(y_test, ls='-', label='Test')
ax.plot(pred_moirai,  ls='--', label='moirai')

ax.set_xlabel('date time')
ax.set_ylabel('Co2 emission')
ax.legend(loc='best')

fig.autofmt_xdate()
plt.tight_layout()
plt.xlim(34100, 34400)
plt.ylim(0, 0.45)
plt.show()




# Do imports
from sktime.forecasting.ttm import TinyTimeMixerForecaster
from sktime.forecasting.chronos import ChronosForecaster

# Initialise models
chronos = ChronosForecaster("amazon/chronos-t5-tiny")
ttm = TinyTimeMixerForecaster()

# Fit
chronos.fit(y=y, fh=range(1, 96))
ttm.fit(y=y, fh=range(1, 96))

# Predict
pred_chronos = chronos.predict(y=y2[:-168])
pred_ttm = ttm.predict(y=y2[:-168])