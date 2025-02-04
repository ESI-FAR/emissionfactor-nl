"""Functions for retrieving data (forecast & historical) through the NED API."""
import datetime
import json
import os
from typing import Literal
import pandas as pd
import requests


def _get_key() -> str:
    key = os.environ.get("NED_API_KEY")
    if key is None:
        msg = "`NED_API_KEY` not found in environment variables."
        raise ValueError(msg)
    return key


TYPE_CODES = {
    "sun": 2,
    "sea-wind": 17,
    "land-wind": 1,
    "mix": 27,
}

URL = "https://api.ned.nl/v1/utilizations"

HEADERS = {
    "X-AUTH-TOKEN": _get_key(),
    "accept": "application/ld+json",
}

DATE_FORMAT = "%Y-%m-%d"
RUNUP_PERIOD = 4*7 # 4 weeks


def _get_last_page(response: requests.Response) -> int:
    """Retrieve the last page nr, as data can be split over multiple pages."""
    json_dict = json.loads(response.text)
    return int(json_dict["hydra:view"]["hydra:last"].split("&page=")[-1])


def _request_data(
    start_date: str,
    end_date: str,
    forecast: bool,
    which: Literal["mix", "sun", "sea-wind", "land-wind"],
    page: int = 1,
) -> requests.Response:
    params = {
        "point": 0,  # NL
        "type": TYPE_CODES[which],
        "granularity": 5,  # Hourly
        "granularitytimezone": 0,  # UTC
        "classification": 1 if forecast else 2,  # historic=2, forecast=1
        "activity": 1,  # Providing
        "validfrom[after]": start_date,  # from (including)
        "validfrom[strictly_before]": end_date,  # up to (excluding)
        "page": page,
    }
    response = requests.get(
        URL, headers=HEADERS, params=params, allow_redirects=False, timeout=60,
    )
    if response.status_code != 200:  # noqa: PLR2004
        msg = (
            f"Error retrieving data from api.ned.nl. Status code {response.status_code}"
        )
        raise requests.ConnectionError(msg)
    return response


def _parse_response(
    response: requests.Response, which: Literal["mix", "sun", "land-wind", "sea-wind"],
) -> pd.DataFrame:
    json_dict = json.loads(response.text)
    if which == "mix":
        vol = []
        ef = []
        dtime = []
        for el in json_dict["hydra:member"]:
            vol.append(float(el["volume"]))
            ef.append(el["emissionfactor"])
            dtime.append(pd.Timestamp(el["validfrom"]))
        df = pd.DataFrame(
            data={"total_volume": vol, "emissionfactor": ef},
            index=dtime,
        )
    else:
        vol = []
        dtime = []
        for el in json_dict["hydra:member"]:
            vol.append(float(el["volume"]))
            dtime.append(pd.Timestamp(el["validfrom"]))
        df = pd.DataFrame(data={f"volume_{which}": vol}, index=dtime)

    df.index = df.index.strftime("%Y-%m-%d %H:%M:%S")
    df.index.name = "time"
    return df


def _get_data(
    sources: tuple[str], start_date: str, end_date: str, forecast: bool,
) -> pd.DataFrame:
    dfs = {source: [] for source in sources}
    for source in sources:
        response = _request_data(
            start_date,
            end_date,
            forecast=forecast,
            which=source,
            page=1,
        )
        dfs[source].append(_parse_response(response, source))

        # Requests >200 items will have multiple pages. Retrieve and append these.
        last_page = _get_last_page(response)
        if last_page >= 2:  # noqa: PLR2004
            for page in range(2, last_page + 1):
                response = _request_data(
                    start_date,
                    end_date,
                    forecast=forecast,
                    which=source,
                    page=page,
                )
                dfs[source].append(_parse_response(response, source))
    return pd.concat(
        [pd.concat(page, axis=0) for page in dfs.values()],
        axis=1,
    )


def get_current_forecast() -> pd.DataFrame:
    """Get the most recent forecast from NED.nl.

    Returns:
        DataFrame containing the forecasted solar, wind and offshore wind production.
    """
    now = datetime.datetime.now(datetime.timezone.utc)
    start_forecast = now.strftime(DATE_FORMAT)
    end_forecast = (now + datetime.timedelta(days=7)).strftime(DATE_FORMAT)

    sources = ("sun", "land-wind", "sea-wind")
    return _get_data(sources, start_forecast, end_forecast, forecast=True)


def get_runup_data() -> pd.DataFrame:
    """Get the historical data from NED.nl, from four weeks ago up to today.

    Returns:
        DataFrame containing the total produced energy, the grid emission factor,
            and the produced solar, wind and offshore wind energy.
    """
    now = datetime.datetime.now(datetime.timezone.utc)
    start_runup = (now - datetime.timedelta(days=RUNUP_PERIOD)).strftime(DATE_FORMAT)
    end_runup = now.strftime(DATE_FORMAT)

    sources = ("mix", "sun", "land-wind", "sea-wind")
    return _get_data(sources, start_runup, end_runup, forecast=False)
