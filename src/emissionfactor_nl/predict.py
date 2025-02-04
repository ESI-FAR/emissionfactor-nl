import os
from pathlib import Path
from autogluon.timeseries import TimeSeriesPredictor
from emissionfactor_nl import retrieve_ned
from emissionfactor_nl.train_model import gluonify

if __name__ == "__main__":
    for env_var in ("NED_API_KEY", "MODEL_PATH", "OUTPUT_PATH"):
        if os.environ.get(env_var) is None:
            msg = f"Environment variable `{env_var}` not set."
            raise ValueError(msg)

    output_path = Path(os.environ.get("OUTPUT_PATH"))
    if not output_path.exists() or not output_path.is_dir():
        msg = (
            "Predictions and forecast data need to be written to a directory\n"
            f"'{output_path}' mounted to this container. E.g.;\n"
            "   docker run -e NED_API_KEY --volume /local/path/to/output/dir:/data"
        )
        raise NotADirectoryError(msg)

    gluon_runup = gluonify(retrieve_ned.get_runup_data())
    gluon_forecast = gluonify(retrieve_ned.get_current_forecast())

    predictor = TimeSeriesPredictor.load(os.environ.get("MODEL_PATH"))
    prediction = predictor.predict(gluon_runup, gluon_forecast)

    date = prediction.index[0][1].strftime("%Y-%m-%d")

    prediction.to_csv(output_path / f"prediction_{date}.csv")
    gluon_runup.to_csv(output_path / f"runup_data_{date}.csv")
    gluon_forecast.to_csv(output_path / f"ned_forecast_{date}.csv")
