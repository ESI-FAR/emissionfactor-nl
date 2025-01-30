import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sktime.forecasting.tbats import TBATS
from sktime.forecasting.bats import BATS
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict
import logging
import time
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CO2Forecaster:
    """A class to handle CO2 emissions forecasting using TBATS and LSTM models."""
    
    def __init__(self, 
                 seq_length: int = 24*7,  # One week of hourly data
                 lstm_units: List[int] = [50, 30],
                 seasonal_periods: List[int] = [24, 24*7],  # Daily and weekly seasonality
                 tbats_use_box_cox: bool = False,
                 tbats_use_trend: bool = True,
                 tbats_use_damped_trend: bool = False,
                 model_save_path: str = 'best_model.keras'):
                 
        """
        Initialize the forecaster with specific parameters.
        
        Args:
            seq_length: Length of sequences for LSTM training
            lstm_units: List of units for LSTM layers
            seasonal_periods: List of seasonal periods for TBATS
            tbats_use_box_cox: Whether to use Box-Cox transformation in TBATS
            tbats_use_trend: Whether to use trend in TBATS
            tbats_use_damped_trend: Whether to use damped trend in TBATS
        """
        self.seq_length = seq_length
        self.lstm_units = lstm_units
        self.seasonal_periods = seasonal_periods
        self.tbats_use_box_cox = tbats_use_box_cox
        self.tbats_use_trend = tbats_use_trend
        self.tbats_use_damped_trend = tbats_use_damped_trend
        self.model_save_path = model_save_path
        self.scaler = MinMaxScaler()
        self.tbats_model = None
        self.lstm_model = None        
        
    
    def read_csv_files_from_folder(self, folder_path):
        """
        Reads all CSV files from a specified folder and returns a concatenated DataFrame.
        """
        dataframes = []
        for filename in os.listdir(folder_path):
            if filename.endswith('.csv'):  
                file_path = os.path.join(folder_path, filename)
                try:
                    df = pd.read_csv(file_path)
                    dataframes.append(df)
                except Exception as e:
                    print(f"Error reading {filename}: {e}")
        
        if dataframes:
            combined_df = pd.concat(dataframes, ignore_index=True)
            return combined_df
        else:
            return pd.DataFrame()

    def preprocess_real_data(self, raw_df, start_date: datetime = None, train_weeks: int = 8, test_weeks: int = 1):
        """
        Preprocess the real data to prepare it for forecasting. This includes:
        - Renaming columns
        - Dropping unnecessary columns
        - Resampling the data to hourly frequency
        - Splitting data into training and testing sets based on specified weeks
        """
        # Preprocess the data
        energy_df = raw_df.rename(columns={'validto (UTC)': 'date_time', 
                                           'emissionfactor (kg CO2/kWh)': 'co2_emissions'}) \
                         .drop(columns=['validfrom (UTC)', 'point', 'granularity', 'percentage', 
                                        'type', 'timezone', 'emission (kg CO2)', 'activity', 
                                        'classification', 'capacity (kW)', 'volume (kWh)']) \
                         .dropna()

        energy_df['date_time'] = pd.to_datetime(energy_df['date_time'])
        energy_df = energy_df.set_index('date_time')
        energy_df = energy_df.resample('H').mean().interpolate()

        # Set the start date if not provided
        if start_date is None:
            start_date = energy_df.index.min()

        # Filter data starting from the specified start date
        energy_df = energy_df[energy_df.index >= start_date]
    
        # Calculate total hours needed        
        total_hours = 24 * 7 * (train_weeks + test_weeks)
        
        # Ensure we have enough data to cover the entire requested range
        if len(energy_df) < total_hours:
            raise ValueError(f"Insufficient data to cover the requested {train_weeks + test_weeks} weeks starting from {start_date}")

        # Take only the required amount of data starting from start_date
        energy_df = energy_df[:total_hours]
    
        # Split the data into training and testing based on hours
        train_size = 24 * 7 * train_weeks
        train_data = energy_df[:train_size]
        test_data = energy_df[train_size:train_size + 24 * 7 * test_weeks]

        # Check variance ratio
        train_var = train_data['co2_emissions'].var()
        test_var = test_data['co2_emissions'].var()
        variance_ratio = max(train_var, test_var) / min(train_var, test_var)
        
        print(f"Train variance: {train_var:.2f}")
        print(f"Test variance: {test_var:.2f}")
        print(f"Variance ratio: {variance_ratio:.2f}")
        print(f"Training data range: {train_data.index.min()} to {train_data.index.max()}")
        print(f"Testing data range: {test_data.index.min()} to {test_data.index.max()}")
        
        # Warning if variance ratio is too high
        if variance_ratio > 1.5:
            print("Warning: Large difference in variance between train and test sets")
        
        return train_data, test_data
   
    
    def create_synthetic_data(self, start_date: datetime = None, 
                            train_weeks: int = 8,  # Reduced to 8 weeks
                            test_weeks: int = 1) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create synthetic hourly CO2 emission data with multiple seasonalities.
        
        Args:
            start_date: Starting date for the data
            train_weeks: Number of weeks for training data
            test_weeks: Number of weeks for test data
            
        Returns:
            Tuple of training and test DataFrames
        """
        if start_date is None:
            start_date = datetime(2024, 1, 1)
            
        total_hours = 24 * 7 * (train_weeks + test_weeks)
        dates = [start_date + timedelta(hours=x) for x in range(total_hours)]
        
        # Create base patterns
        hourly_pattern = 10 * np.sin(np.linspace(0, 2*np.pi, 24))
        weekly_pattern = 15 * np.sin(np.linspace(0, 2*np.pi, 24*7))
        
        # Add working days pattern (Mon-Fri higher than Sat-Sun)
        workday_pattern = np.array([1.2] * 120 + [0.8] * 48) * 15  # 5 days high, 2 days low
        
        co2_values = []
        for i in range(total_hours):
            hour_of_day = i % 24
            day_of_week = (i // 24) % 7
            
            value = (
                100 +
                hourly_pattern[hour_of_day] * (1.2 if 8 <= hour_of_day <= 17 else 0.8) +
                weekly_pattern[i % (24*7)] * 0.8 +
                workday_pattern[i % len(workday_pattern)]
            )
            
            # Add controlled noise to maintain similar variance
            noise = np.random.normal(0, 2)
            co2_values.append(max(0, value + noise))
        
        df = pd.DataFrame({
            'timestamp': dates,
            'co2_emissions': co2_values
        })
        df.set_index('timestamp', inplace=True)
        
        # Split into train and test
        train_size = 24 * 7 * train_weeks
        train_data = df[:train_size]
        test_data = df[train_size:]

        # Check variance ratio
        train_var = train_data['co2_emissions'].var()
        test_var = test_data['co2_emissions'].var()
        variance_ratio = max(train_var, test_var) / min(train_var, test_var)
        
        logger.info(f"Train variance: {train_var:.2f}")
        logger.info(f"Test variance: {test_var:.2f}")
        logger.info(f"Variance ratio: {variance_ratio:.2f}")
        
        # Warning if variance ratio is too high
        if variance_ratio > 1.5:  # threshold can be adjusted
            logger.warning("Warning: Large difference in variance between train and test sets")
        
        return train_data, test_data
    
    def create_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM training with overlap."""
        sequences, targets = [], []
        for i in range(len(data) - self.seq_length):
            sequences.append(data[i:(i + self.seq_length)])
            targets.append(data[i + self.seq_length])
        return np.array(sequences), np.array(targets)
    
    def build_lstm_model(self) -> Sequential:
        """Build an improved LSTM model with dropout and better architecture."""
        model = Sequential([
            LSTM(self.lstm_units[0], activation='relu', 
                 input_shape=(self.seq_length, 1), return_sequences=True),
            Dropout(0.2),
            LSTM(self.lstm_units[1], activation='relu'),
            Dropout(0.2),
            Dense(16, activation='relu'),  # Reduced dense layer size
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='huber')
        return model
    
    
    def check_data_statistics(self, train_data: pd.DataFrame, 
                            test_data: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """
        Check various statistical properties of the datasets.
        
        Returns:
            Dictionary containing statistics for both datasets
        """
        stats = {}
        
        for name, data in [('train', train_data), ('test', test_data)]:
            stats[name] = {
                'mean': data['co2_emissions'].mean(),
                'std': data['co2_emissions'].std(),
                'variance': data['co2_emissions'].var(),
                'min': data['co2_emissions'].min(),
                'max': data['co2_emissions'].max(),
                'skew': data['co2_emissions'].skew(),
                'kurtosis': data['co2_emissions'].kurtosis()
            }
        
        # Calculate comparative metrics
        stats['comparative'] = {
            'variance_ratio': stats['train']['variance'] / stats['test']['variance'],
            'mean_difference': abs(stats['train']['mean'] - stats['test']['mean']),
            'std_ratio': stats['train']['std'] / stats['test']['std']
        }
        
        return stats
    

    def fit(self, train_data: pd.DataFrame, test_data: pd.DataFrame) -> Dict[str, float]:
        """
        Fit both TBATS and LSTM models after checking data statistics.
        """
        # Check data statistics before fitting
        stats = self.check_data_statistics(train_data, test_data)
        
        # Log important statistics
        logger.info("\nData Statistics:")
        logger.info(f"Variance ratio (train/test): {stats['comparative']['variance_ratio']:.2f}")
        logger.info(f"Mean difference: {stats['comparative']['mean_difference']:.2f}")
        logger.info(f"Standard deviation ratio: {stats['comparative']['std_ratio']:.2f}")
        
        # Proceed with fitting if statistics are acceptable
        if stats['comparative']['variance_ratio'] > 1.5 or stats['comparative']['variance_ratio'] < 0.67:
            logger.warning("Warning: Significant difference in variance between train and test sets")
        
        training_times = {}
        
        # Fit TBATS
        logger.info("Fitting TBATS model...")
        start_time = time.time()
        self.tbats_model = BATS(
            use_box_cox=self.tbats_use_box_cox,
            use_trend=self.tbats_use_trend,
            use_damped_trend=self.tbats_use_damped_trend,
            sp=self.seasonal_periods,
            use_arma_errors=False,
            n_jobs=-1
            )
        
        self.tbats_model.fit(train_data['co2_emissions'])
        training_times['tbats'] = time.time() - start_time
        
        # Fit LSTM
        logger.info("Preparing data for LSTM...")
        start_time = time.time()
        scaled_train = self.scaler.fit_transform(train_data['co2_emissions'].values.reshape(-1, 1))
        X_train, y_train = self.create_sequences(scaled_train)
        
        logger.info("Fitting LSTM model...")
        self.lstm_model = self.build_lstm_model()
        
        # Updated callbacks with correct file extension
        callbacks = [
            EarlyStopping(
                monitor='val_loss',          # Monitor the validation loss
                patience=5,                  # Stop if no improvement for 5 epochs
                restore_best_weights=True,   # Restore the weights of the best epoch
                mode='min'                   # Minimize the validation loss
            ),
            ModelCheckpoint(
                filepath=self.model_save_path,   # Save the model to a file
                monitor='val_loss',              # Monitor the validation loss
                save_best_only=True,             # Save only the best model
                verbose=1                        # Print progress messages
            )
        ]
        
        history = self.lstm_model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=1
        )
        training_times['lstm'] = time.time() - start_time
        
        return training_times
    
    def predict(self, test_data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate predictions from both models and combine them."""
        predictions = {}
        
        # TBATS forecast
        logger.info("Generating TBATS forecast...")
        predictions['tbats'] = self.tbats_model.predict(fh=np.arange(1, len(test_data) + 1))
        
        # LSTM forecast
        logger.info("Generating LSTM forecast...")
        scaled_test = self.scaler.transform(test_data['co2_emissions'].values.reshape(-1, 1))
        last_sequence = scaled_test[:self.seq_length]
        lstm_forecast = []
        
        for _ in range(len(test_data)):
            next_pred = self.lstm_model.predict(last_sequence.reshape(1, self.seq_length, 1), verbose=0)
            lstm_forecast.append(next_pred[0])
            last_sequence = np.append(last_sequence[1:], next_pred)
        
        predictions['lstm'] = self.scaler.inverse_transform(np.array(lstm_forecast)).flatten()
        
        # Adaptive weighted combination based on recent performance
        recent_errors_tbats = np.abs(predictions['tbats'][:24] - test_data['co2_emissions'].values[:24])
        recent_errors_lstm = np.abs(predictions['lstm'][:24] - test_data['co2_emissions'].values[:24])
        
        tbats_weight = 1 / (np.mean(recent_errors_tbats) + 1e-6)
        lstm_weight = 1 / (np.mean(recent_errors_lstm) + 1e-6)
        total_weight = tbats_weight + lstm_weight
        
        predictions['combined'] = (
            (tbats_weight * predictions['tbats'] + lstm_weight * predictions['lstm']) / total_weight
        )
        
        return predictions
    
    def evaluate(self, test_data: pd.DataFrame, predictions: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
        """Evaluate the models using multiple metrics."""
        metrics = {}
        actual = test_data['co2_emissions'].values
        
        for model_name, preds in predictions.items():
            metrics[model_name] = {
                'rmse': np.sqrt(mean_squared_error(actual, preds)),
                'mae': mean_absolute_error(actual, preds),
                'r2': r2_score(actual, preds),
                'mape': np.mean(np.abs((actual - preds) / actual)) * 100
            }
        
        return metrics
    
    def plot_results(self, test_data: pd.DataFrame, predictions: Dict[str, np.ndarray]) -> None:
        """Plot the forecasting results."""
        plt.figure(figsize=(12, 6))
        plt.plot(test_data.index, test_data['co2_emissions'], label='Actual', marker='o', markersize=4)
        
        colors = {'tbats': 'red', 'lstm': 'green', 'combined': 'blue'}
        for model_name, preds in predictions.items():
            plt.plot(test_data.index, preds, label=f'{model_name.upper()}',
                    linestyle='--', color=colors[model_name])
        
        plt.title('CO2 Emissions Forecasting Results')
        plt.xlabel('Timestamp')
        plt.ylabel('CO2 Emissions')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

#     Print statistical comparison
#     print("\nData Statistics Comparison:")
#     print(f"Train variance: {stats['train']['variance']:.2f}")
#     print(f"Test variance: {stats['test']['variance']:.2f}")
#     print(f"Variance ratio: {stats['comparative']['variance_ratio']:.2f}")
#     print(f"Mean difference: {stats['comparative']['mean_difference']:.2f}")
#     print(f"Std ratio: {stats['comparative']['std_ratio']:.2f}")

