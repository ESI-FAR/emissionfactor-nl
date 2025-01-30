from datetime import datetime
from co2_forecaster import CO2Forecaster

def main(use_real_data = False, data_folder = '../Data/electric/'):
    """Main function to test the CO2Forecaster"""
    try:
        # Initialize forecaster
        forecaster = CO2Forecaster(
            seq_length=24*7,
            lstm_units=[50, 30],
            seasonal_periods=[24, 24*7],
            model_save_path='best_model.keras'
        )
        
        if use_real_data:
            print("Loading real data from folder...")
            raw_data = forecaster.read_csv_files_from_folder(data_folder)
            train_data, test_data = forecaster.preprocess_real_data(
                raw_data,
                start_date=datetime(2024, 9, 15),  # select when to start the training data
                train_weeks=10,
                test_weeks=1
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
        
        # Plot and compare results of BATS, LSTM and hybrid models
        print("\nPlotting results...")
        forecaster.plot_results(test_data, predictions)
        
        return forecaster, train_data, test_data, predictions, metrics, training_times
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    # Change the flag 'use_real_data' below to toggle between synthetic or real data
    main(use_real_data = True, data_folder = '../Data/electric/')