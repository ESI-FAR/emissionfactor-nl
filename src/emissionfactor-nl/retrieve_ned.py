import datetime
import json
import os
from typing import Literal
import pandas
import requests


def get_key() -> str:
    key = os.environ.get("ned_api_key")
    if key is None:
        msg = "`ned_api_key` not found in environment variables."
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
    "X-AUTH-TOKEN": get_key(),
    "accept": "application/ld+json",
}

DATE_FORMAT = "%Y-%m-%d"
RUNUP_PERIOD = 4*7 # 4 weeks


def get_last_page(response: requests.Response) -> int:
    """Retrieve the last page nr, as data can be split over multiple pages."""
    json_dict = json.loads(response.text)
    return int(json_dict['hydra:view']['hydra:last'].split("&page=")[-1])


def request_data(
    start_date: str,
    end_date: str,
    forecast: bool,
    which: Literal["mix", "sun", "sea-wind", "land-wind"],
    page: int = 1,
):
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
    response = requests.get(URL, headers=HEADERS, params=params, allow_redirects=False)
    if response.status_code != 200:
        msg = f"Error retrieving data from api.ned.nl. Status code {response.status_code}"
        raise requests.ConnectionError(msg)
    return response


def parse_response(
    response: requests.Response, which: Literal["mix", "sun", "land-wind", "sea-wind"]
):
    json_dict = json.loads(response.text)
    if which == "mix":
        vol = list()
        ef = list()
        dtime = list()
        for el in json_dict['hydra:member']:
            vol.append(float(el["volume"]))
            ef.append(el["emissionfactor"])
            dtime.append(pandas.Timestamp(el["validfrom"]))
        df = pandas.DataFrame(
            data={"total_volume": vol, "emissionfactor": ef},
            index=dtime
        )
    else:
        vol = list()
        dtime = list()
        for el in json_dict['hydra:member']:
            vol.append(float(el["volume"]))
            dtime.append(pandas.Timestamp(el["validfrom"]))
        df = pandas.DataFrame(data={f"volume_{which}": vol}, index=dtime)

    df.index = df.index.strftime("%Y-%m-%d %H:%M:%S")
    df.index.name = "time"
    return df


def get_data(
    sources: tuple[str], start_date: str, end_date: str, forecast: bool
) -> pandas.DataFrame:
    dfs = {source: [] for source in sources}
    for source in sources:
        response = request_data(
            start_date,
            end_date,
            forecast=forecast,
            which=source,
            page=1,
        )
        dfs[source].append(parse_response(response, source))

        # Requests >200 items will have multiple pages. Retrieve and append these.
        last_page = get_last_page(response)
        if last_page >=2 :
            for page in range(2, last_page + 1):
                response = request_data(
                    start_date,
                    end_date,
                    forecast=forecast,
                    which=source,
                    page=page,
                )
                dfs[source].append(parse_response(response, source))
    return pandas.concat(
        [pandas.concat(page, axis=0) for page in dfs.values()],
        axis=1,
    )


def get_current_forecast() -> pandas.DataFrame:
    now = datetime.datetime.now()
    start_forecast = now.strftime(DATE_FORMAT)
    end_forecast = (now + datetime.timedelta(days=7)).strftime(DATE_FORMAT)

    sources = ("sun", "land-wind", "sea-wind")
    return get_data(sources, start_forecast, end_forecast, forecast=True)


def get_runup_data() -> pandas.DataFrame:
    now = datetime.datetime.now()
    start_runup = (now - datetime.timedelta(days=RUNUP_PERIOD)).strftime(DATE_FORMAT)
    end_runup = now.strftime(DATE_FORMAT)

    sources = ("mix", "sun", "land-wind", "sea-wind")
    return get_data(sources, start_runup, end_runup, forecast=False)
