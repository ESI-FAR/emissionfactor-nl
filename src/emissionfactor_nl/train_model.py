import os
from pathlib import Path
import numpy as np
import pandas as pd
from autogluon.timeseries import TimeSeriesDataFrame
from autogluon.timeseries import TimeSeriesPredictor
from emissionfactor_nl import read_ned
from emissionfactor_nl.read_knmi import parse_knmi_uurgeg


def gluonify(df: pd.DataFrame) -> TimeSeriesDataFrame:
    """Convert a pandas dataframe to the format expected by autogluon."""
    df = df.reset_index()
    df["item_id"] = 0
    return TimeSeriesDataFrame.from_data_frame(df, timestamp_column="time")


PREDICTOR_LENGTH = 7 * 24
FRIDAY = 4


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

    knmi_weather_file = Path(training_data_path) / "uurgeg_260_2021-2030.txt"
    df_knmi = parse_knmi_uurgeg(knmi_weather_file)

    # Combine data sources
    data = ned_data.join(df_knmi)

    data["datetime"] = pd.DatetimeIndex(data.index)
    data["weekend"] = data["datetime"].dt.day_of_week > FRIDAY
    data["volume_wind"] = data["volume_land-wind"] + data["volume_sea-wind"]
    data["volume_green"] = data["volume_sun"] + data["volume_wind"]

    # Strip out test data
    train_data = data[:-PREDICTOR_LENGTH]
    test_data = data[-PREDICTOR_LENGTH:]

    gluon_train_data = gluonify(train_data)
    gluon_test_data = gluonify(test_data)

    predictor = TimeSeriesPredictor(
        prediction_length=PREDICTOR_LENGTH,
        freq="1h",
        target="emissionfactor",
        known_covariates_names=["volume_green", "weekend", "air_temperature"],
        path=model_path,
    ).fit(
        gluon_train_data,
        excluded_model_types=["Chronos", "DeepAR", "TiDE"],
    )
