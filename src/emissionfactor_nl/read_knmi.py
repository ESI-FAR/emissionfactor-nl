import pandas as pd


def parse_knmi_uurgeg(file: str) -> pd.DataFrame:
    """Read KNMI observed air temperature from file.

    Args:
        file: filename of the KNMI data
    """
    df_knmi = pd.read_csv(
        file, skip_blank_lines=True, skiprows=31, skipinitialspace=True
    )
    df_knmi["time"] = [
        f"{_date[:4]}-{_date[4:6]}-{_date[6:]} {_time-1:02}:00:00"
        for _date, _time in zip(df_knmi["YYYYMMDD"].astype(str), df_knmi["HH"])  # noqa: B905
    ]
    df_knmi["air_temperature"] = df_knmi["T"].astype(float)/10

    df_knmi = df_knmi.set_index("time")
    return df_knmi[["air_temperature"]]
