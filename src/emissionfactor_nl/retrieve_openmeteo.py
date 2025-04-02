import os
import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry

OPENMETEO_URL = "https://api.open-meteo.com/v1/forecast"
LOCATION = {
    "latitude": 52.15, # Amersfoort (center of NL)
    "longitude": 5.39,
}


def get_openmeteo_client() -> openmeteo_requests.Client:
    """Setup the Open-Meteo API client with cache and retry on error."""
    cache_session = requests_cache.CachedSession(".cache", expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    return openmeteo_requests.Client(session=retry_session)


def retrieve_temperature() -> pd.DataFrame:
    """Get the 2m air temperature forecast & runup period for the center of the NL."""
    params = {
        **LOCATION,
        "hourly": "temperature_2m",
        "models": "knmi_seamless",
        "forecast_days": 10,
        "past_days": 28,
    }

    # Use API key if available
    apikey = os.environ.get("OPENMETEO_API_KEY")
    if apikey is not None:
        params["apikey"] = apikey

    responses = get_openmeteo_client().weather_api(OPENMETEO_URL, params=params)
    response = responses[0]  # only one location/model

    hourly = response.Hourly()
    hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()

    hourly_data = {"date": pd.date_range(
        start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
        end = pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
        freq = pd.Timedelta(seconds = hourly.Interval()),
        inclusive = "left"
    )}

    hourly_data["air_temperature"] = hourly_temperature_2m
    df_hourly = pd.DataFrame(data = hourly_data)
    df_hourly["date"] = pd.DatetimeIndex(df_hourly["date"]).tz_localize(None)
    return df_hourly.set_index("date")
