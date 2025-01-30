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
from typing import Tuple, List, Dict, Set
import logging
import time
import os
from statsmodels.tsa.statespace.sarimax import SARIMAX
from itertools import combinations

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CO2Forecaster:
    """A class to handle univariate CO2 emissions forecasting using a combination of TBATS and LSTM models."""
    
    def __init__(self, 
                 models: List[str] = ['tbats', 'lstm', 'sarima'],
                 sequential: bool = False,
                 seq_length: int = 24*7,
                 lstm_units: List[int] = [50, 30],
                 seasonal_periods: List[int] = [24, 24*7],
                 tbats_use_box_cox: bool = False,
                 tbats_use_trend: bool = True,
                 tbats_use_damped_trend: bool = True,
                 sarima_order: tuple = (1, 1, 1),
                 sarima_seasonal_order: tuple = (1, 1, 1, 24),
                 model_save_path: str = 'best_model.keras'):
                 
        """
        Initialize the forecaster with specific parameters.
        
        Args:
            models: List of models to use ('tbats', 'lstm', 'sarima')
            sequential: If True, models are run sequentially on residuals
            seq_length: Length of sequences for LSTM training
            lstm_units: List of units for LSTM layers
            seasonal_periods: List of seasonal periods for TBATS
            sarima_order: SARIMA (p,d,q) parameters
            sarima_seasonal_order: SARIMA seasonal (P,D,Q,s) parameters
        """
        
        self.models = models
        self.sequential = sequential
        self.seq_length = seq_length
        self.lstm_units = lstm_units
        self.seasonal_periods = seasonal_periods
        self.tbats_use_box_cox = tbats_use_box_cox
        self.tbats_use_trend = tbats_use_trend
        self.tbats_use_damped_trend = tbats_use_damped_trend
        self.sarima_order = sarima_order
        self.sarima_seasonal_order = sarima_seasonal_order
        self.model_save_path = model_save_path
        self.scaler = MinMaxScaler()
        
        # Initialize model objects
        self.model_objects = {
            'tbats': None,
            'lstm': None,
            'sarima': None
        }
    
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

    def preprocess_real_data(self, raw_df, start_date: datetime = None, train_weeks: int = 4, test_weeks: int = 1):
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
        #energy_df = energy_df.resample('H').mean().interpolate()

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
                            train_weeks: int = 4,  # Reduced to 8 weeks
                            test_weeks: int = 1) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create synthetic hourly CO2 emission data with multiple seasonalities
        
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
        """Create sequences for LSTM training with overlap"""
        sequences, targets = [], []
        for i in range(len(data) - self.seq_length):
            sequences.append(data[i:(i + self.seq_length)])
            targets.append(data[i + self.seq_length])
        return np.array(sequences), np.array(targets)
    
    def build_lstm_model(self) -> Sequential:
        """Build LSTM model"""
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
    
    def fit_sarima(self, train_data: pd.DataFrame) -> tuple:
        """Fit SARIMA model to the data."""
        start_time = time.time()
        
        model = SARIMAX(train_data['co2_emissions'],
                       order=self.sarima_order,
                       seasonal_order=self.sarima_seasonal_order)
        
        self.model_objects['sarima'] = model.fit(disp=False)
        training_time = time.time() - start_time
        
        return training_time
    

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
        
        # Sequential mode: TBATS + LSTM/SARIMA on the residuals of TBATS
        if self.sequential:
            residuals = train_data.copy()
            for model in self.models:
                if model == 'tbats':
                    self.model_objects['tbats'] = TBATS(
                        use_box_cox      = self.tbats_use_box_cox,
                        use_trend        = self.tbats_use_trend,
                        use_damped_trend = self.tbats_use_damped_trend,
                        sp               = self.seasonal_periods,
                        use_arma_errors  = True,
                        n_jobs           = -1
                    )
                    training_times['tbats'] = time.time()
                    self.model_objects['tbats'].fit(residuals['co2_emissions'])
                    training_times['tbats'] = time.time() - training_times['tbats']
                    
                    # Calculate residuals for next model
                    if len(self.models) > 1:
                        pred = self.model_objects['tbats'].predict(fh=np.arange(1, len(train_data) + 1))
                        residuals['co2_emissions'] = train_data['co2_emissions'] - pred
                
                elif model == 'lstm':
                    training_times['lstm'] = time.time()
                    scaled_train = self.scaler.fit_transform(residuals['co2_emissions'].values.reshape(-1, 1))
                    X_train, y_train = self.create_sequences(scaled_train)
                    
                    self.model_objects['lstm'] = self.build_lstm_model()
                    callbacks = [
                        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
                        ModelCheckpoint(filepath=self.model_save_path, monitor='val_loss', save_best_only=True)
                    ]
                    
                    self.model_objects['lstm'].fit(
                        X_train, y_train,
                        epochs=50,
                        batch_size=32,
                        validation_split=0.2,
                        callbacks=callbacks,
                        verbose=0
                    )
                    training_times['lstm'] = time.time() - training_times['lstm']
                    
                    if len(self.models) > 1:
                        # Calculate residuals for next model
                        lstm_pred = self.predict_lstm(train_data)
                        residuals['co2_emissions'] = train_data['co2_emissions'] - lstm_pred
                
                elif model == 'sarima':
                    training_times['sarima'] = self.fit_sarima(residuals)
                    if len(self.models) > 1:
                        # Calculate residuals for next model
                        sarima_pred = self.predict_sarima(train_data)
                        residuals['co2_emissions'] = train_data['co2_emissions'] - sarima_pred
        
        else:  # Parallel mode: run TBATS and LSTM (or SARIMA) separately, and later take a combination
            for model in self.models:
                if model == 'tbats':
                    self.model_objects['tbats'] = TBATS(
                        use_box_cox      = self.tbats_use_box_cox,
                        use_trend        = self.tbats_use_trend,
                        use_damped_trend = self.tbats_use_damped_trend,
                        sp               = self.seasonal_periods,
                        use_arma_errors  = True,
                        n_jobs           = -1
                    )
                    training_times['tbats'] = time.time()
                    self.model_objects['tbats'].fit(train_data['co2_emissions'])
                    training_times['tbats'] = time.time() - training_times['tbats']
                
                elif model == 'lstm':
                    training_times['lstm'] = time.time()
                    scaled_train = self.scaler.fit_transform(train_data['co2_emissions'].values.reshape(-1, 1))
                    X_train, y_train = self.create_sequences(scaled_train)
                    
                    self.model_objects['lstm'] = self.build_lstm_model()
                    callbacks = [
                        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
                        ModelCheckpoint(filepath=self.model_save_path, monitor='val_loss', save_best_only=True)
                    ]
                    
                    self.model_objects['lstm'].fit(
                        X_train, y_train,
                        epochs=50,
                        batch_size=32,
                        validation_split=0.2,
                        callbacks=callbacks,
                        verbose=0
                    )
                    training_times['lstm'] = time.time() - training_times['lstm']
                
                elif model == 'sarima':
                    training_times['sarima'] = self.fit_sarima(train_data)
        
        return training_times
    
    def predict_sarima(self, test_data: pd.DataFrame) -> np.ndarray:
        """Generate predictions from SARIMA model."""
        return self.model_objects['sarima'].forecast(len(test_data))

    def predict_lstm(self, test_data: pd.DataFrame) -> np.ndarray:
        """Generate predictions from LSTM model."""
        scaled_test = self.scaler.transform(test_data['co2_emissions'].values.reshape(-1, 1))
        last_sequence = scaled_test[:self.seq_length]
        lstm_forecast = []
        
        for _ in range(len(test_data)):
            next_pred = self.model_objects['lstm'].predict(last_sequence.reshape(1, self.seq_length, 1), verbose=0)
            lstm_forecast.append(next_pred[0])
            last_sequence = np.append(last_sequence[1:], next_pred)
        
        return self.scaler.inverse_transform(np.array(lstm_forecast)).flatten()

    def predict(self, test_data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate predictions from selected models."""
        predictions = {}
        
        # Sequential mode: TBATS + LSTM or SARIMA on the residuals of TBATS
        if self.sequential:
            residuals = test_data.copy()
            combined_pred = np.zeros(len(test_data))
            
            for model in self.models:
                if model == 'tbats':
                    pred = self.model_objects['tbats'].predict(fh=np.arange(1, len(test_data) + 1))
                    predictions['tbats'] = pred
                    combined_pred += pred
                    if len(self.models) > 1:
                        residuals['co2_emissions'] = test_data['co2_emissions'] - pred
                
                elif model == 'lstm':
                    pred = self.predict_lstm(residuals)
                    predictions['lstm'] = pred
                    combined_pred += pred                    
                
                elif model == 'sarima':
                    pred = self.predict_sarima(residuals)
                    predictions['sarima'] = pred
                    combined_pred += pred
            
            predictions['combined'] = combined_pred
        
        else:  # Parallel mode: prediction based on weighted combination of TBATS + LSTM (or SARIMA)
            model_preds = {}
            for model in self.models:
                if model == 'tbats':
                    model_preds['tbats']  = self.model_objects['tbats'].predict(fh=np.arange(1, len(test_data) + 1))
                elif model == 'lstm':
                    model_preds['lstm']   = self.predict_lstm(test_data)
                elif model == 'sarima':
                    model_preds['sarima'] = self.predict_sarima(test_data)
            
            # Store individual predictions
            predictions.update(model_preds)
            
            # Calculate adaptive weights based on recent performance
            if len(self.models) > 1:
                weights = {}
                for model in self.models:
                    recent_errors = np.abs(model_preds[model][:24] - test_data['co2_emissions'].values[:24])
                    weights[model] = 1 / (np.mean(recent_errors) + 1e-6)
                
                total_weight = sum(weights.values())
                predictions['combined'] = np.zeros(len(test_data))
                for model in self.models:
                    predictions['combined'] += (weights[model] / total_weight) * model_preds[model]
        
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
        
        colors = {
        'tbats': '#FF6B6B',    # Red
        'lstm': '#4ECDC4',     # Teal
        'sarima': '#95A5A6',   # Gray
        'combined': '#45B7D1'  # Blue
        }
        
        linestyles = {
        'tbats': '--',
        'lstm': ':',
        'sarima': '-.',
        'combined': '-'
        }
        
        for model_name, preds in predictions.items():
            plt.plot(test_data.index, preds, 
                    label=f'{model_name.upper()}',
                    linestyle=linestyles[model_name],
                    color=colors[model_name])
                
        plt.title('CO2 Emissions Forecasting Results')
        plt.xlabel('Timestamp')
        plt.ylabel('CO2 Emissions')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

def main(use_real_data = False, data_folder = '../Data/electric/'):
    """Main function to test the CO2Forecaster"""
    try:
        # Initialize forecaster
        
        # forecaster = CO2Forecaster(
        #     models=['tbats', 'sarima'],  # Select models
        #     sequential=True,  # True for residual-based sequential mode
        #     sarima_order=(1,1,1),
        #     sarima_seasonal_order=(1,0,1,24)
        # )
        
        forecaster = CO2Forecaster(            
            models=['tbats','lstm'],
            sequential = True ,  # True for residual-based sequential mode
            seq_length=24*7,
            lstm_units=[50, 30],
            seasonal_periods=[24, 24*7],
            model_save_path='best_model.keras'
        )
        
        # load in ned.nl data on co2 emission
        if use_real_data:
            print("Loading real data on co2 emission from folder...")
            raw_data = forecaster.read_csv_files_from_folder(data_folder)
            train_data, test_data = forecaster.preprocess_real_data(
                raw_data,
                start_date=datetime(2024, 10, 15),  # select when to start the training data
                train_weeks = 4,
                test_weeks  = 1
            )
        else:
            # Generate synthetic data
            print("Generating synthetic data...")
            train_data, test_data = forecaster.create_synthetic_data(
                start_date=datetime(2024, 1, 1),
                train_weeks=8,
                test_weeks=1
            )           
                
        # Fit models
        print("Fitting models...")
        training_times = forecaster.fit(train_data, test_data)
        
        # Generate predictions
        print("Generating predictions...")
        predictions = forecaster.predict(test_data)
        
        # Evaluate models
        print("Evaluating models...")
        metrics = forecaster.evaluate(test_data, predictions)
        
        # Print metrics
        print("\nModel Performance Metrics:")
        for model_name, model_metrics in metrics.items():
            print(f"\n{model_name.upper()}:")
            for metric_name, value in model_metrics.items():
                print(f"{metric_name}: {value:.4f}")
        
        # Plot and compare results of TBATS, LSTM and hybrid model
        print("\nPlotting results...")
        forecaster.plot_results(test_data, predictions)
        
        return forecaster, train_data, test_data, predictions, metrics, training_times
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    # Change the flag 'use_real_data' below to toggle between synthetic or real data
    main(use_real_data = True, data_folder = '../Data/electric/')