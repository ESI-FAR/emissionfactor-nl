import os
from pathlib import Path
import pandas as pd
from autogluon.timeseries import TimeSeriesDataFrame
from autogluon.timeseries import TimeSeriesPredictor
from emissionfactor_nl import read_ned


def gluonify(df: pd.DataFrame) -> TimeSeriesDataFrame:
    """Convert a pandas dataframe to the format expected by autogluon."""
    df = df.reset_index()
    df["item_id"] = 0
    return TimeSeriesDataFrame.from_data_frame(df, timestamp_column="time")


if __name__ == "__main__":
    # Load env vars
    training_data_path = os.environ.get("TRAINING_DATA")
    if training_data_path is None:
        raise ValueError

    model_path = os.environ.get("MODEL_PATH")
    if model_path is None:
        raise ValueError

    # Load data
    ned_data = read_ned.read_all(Path(training_data_path))
    ned_data.index = ned_data.index.astype(str)

    train_data = ned_data[:-7*24]
    test_data = ned_data[-7*24:]

    gluon_train_data = gluonify(train_data)
    gluon_test_data = gluonify(test_data)

    predictor = TimeSeriesPredictor(
        prediction_length=7*24,
        freq="1h",
        target="emissionfactor",
        known_covariates_names=["volume_sun", "volume_land-wind", "volume_sea-wind"],
        path=model_path,
    ).fit(
        gluon_train_data,
        excluded_model_types=["Chronos", "DeepAR", "TiDE"],
    )
