"""Functions for reading NED.nl data from their .csv files."""
from collections.abc import Iterable
from pathlib import Path
from typing import Literal
import pandas as pd


def read_predictions(directory: Path) -> pd.DataFrame:
    """Read NED predictions from .csv files.

    Files have to be downloaded from ned.nl, and be put into a single directory.

    Args:
        directory: Directory containing the forecast files.

    Returns:
        Dataframe with the merged solar, wind and offshore wind data.
    """
    df_sun = _read_ned(directory.glob("zon-uur-voorspelling*.csv"), "sun")
    df_wind = _read_ned(directory.glob("wind-uur-voorspelling*.csv"), "land-wind")
    df_wind_sea = _read_ned(directory.glob("zeewind-uur-voorspelling*.csv"), "sea-wind")

    combined_data = df_sun.copy()
    for data in (df_wind, df_wind_sea):
        combined_data = combined_data.merge(data, on="time", how="outer")

    return combined_data


def read_all(directory: Path) -> pd.DataFrame:
    """Read NED historical data from .csv files.

    Files have to be downloaded from ned.nl, and be put into a single directory.

    For example, the following files for 2021 and 2022:
        data/
            electriciteitsmix-2021-uur-data.csv
            electriciteitsmix-2022-uur-data.csv
            wind-2021-uur-data.csv
            wind-2022-uur-data.csv
            zeewind-2021-uur-data.csv
            zeewind-2022-uur-data.csv
            zon-2021-uur-data.csv
            zon-2022-uur-data.csv

    Args:
        directory: Directory containing the forecast files.

    Returns:
        Dataframe with the merged total production, emissionfactor and produced
            solar, wind and offshore wind energy.
    """
    df_mix = _read_ned(directory.glob("electriciteitsmix-*-uur-data.csv"), "mix")
    df_sun = _read_ned(directory.glob("zon-*-uur-data.csv"), "sun")
    df_wind = _read_ned(directory.glob("wind-*-uur-data.csv"), "land-wind")
    df_wind_sea = _read_ned(directory.glob("zeewind-*-uur-data.csv"), "sea-wind")

    combined_data = df_mix.copy()
    for data in (df_sun, df_wind, df_wind_sea):
        combined_data = combined_data.merge(data, on="time", how="outer")

    return combined_data


def _read_ned(
    files: Iterable[Path], which: Literal["mix", "sun", "land-wind", "sea-wind"],
) -> pd.DataFrame:
    data = []
    for file in sorted(files):
        if which == "mix":
            data.append(_read_mix_file(file))
        else:
            data.append(_read_production_file(file, which))
    return pd.concat(data)


def _read_mix_file(fname: str | Path) -> pd.DataFrame:
    df = pd.read_csv(
        fname,
        usecols=("validfrom (UTC)", "volume (kWh)", "emissionfactor (kg CO2/kWh)"),
    )

    df = df.rename(columns={
        "validfrom (UTC)": "time",
        "volume (kWh)": "total_volume",
        "emissionfactor (kg CO2/kWh)": "emissionfactor",
    })
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time")
    df["total_volume"] = df["total_volume"].astype(float)
    return df


def _read_production_file(
    fname: str | Path, which: Literal["sun", "land-wind", "sea-wind"],
) -> pd.DataFrame:
    df = pd.read_csv(
        fname,
        usecols=("validfrom (UTC)", "volume (kWh)"))

    name_vol = f"volume_{which}"
    df = df.rename(columns={
        "validfrom (UTC)": "time",
        "volume (kWh)": name_vol,
    })
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time")
    df[name_vol] = df[name_vol].astype(float)
    return df
