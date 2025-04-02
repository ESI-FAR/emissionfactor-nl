# Forecasting grid emission factor for the Netherlands

This repository contains a workflow to produce emission factor forecasts for the
electricity mix of the Netherlands, up to 7 days ahead.

## Model training
Model training is performed with [AutoGluon](https://auto.gluon.ai/), using the time
series forcasting module.

As training data the total energy production and the energy mix's emission factor are
used, sourced from the [Nationaal Energie Dashboard](https://ned.nl/)
with the produced solar and wind energy as "known covariates".

<img src="img/model_test.png" alt="Model training test result" width="400"/>

*Model training result, validation on unseen data*

The NED also provides forecasts for the solar and wind production.
These are used in forecasting of the emission factor.

<img src="img/example_forecast.png" alt="Example forecast" width="400"/>

*Example forecast, for 2025-01-28 - 2025-02-04*

To produce a better forecast, the model also makes use of forecasted air temperature
data.

## Reproducing results

The Dockerfile contained in this reposity describes all steps you need to go
through to train a model and to produce a forecast.

The container image allows easy production of forecasts.
Docker engine is [freely available on Linux](https://docs.docker.com/engine/install/).

To run the container image do:

```docker
docker run \
    -e NED_API_KEY \
    --volume /local/path/to/output/dir:/data \
    ghcr.io/esi-far/emissionfactor-forecast:0.1.0
```

The `/data` directory is the location where the prediction file should end up.
The container will also write the run-up data used in the prediction, as well as
NED's forecast for available wind and solar energy.

The environmental variable `NED_API_KEY` should be your ned.nl API key. Set this with:
```sh
export NED_API_KEY=enter-your-key-here
```
More information on the NED API is available [here](https://ned.nl/nl/api).

> [!IMPORTANT]
> If you want to use the forecast for commercial use, you will have get a commercial
> license and API key from [OpenMeteo](https://open-meteo.com/). To pass this API key,
> set the `OPENMETEO_API_KEY` environment variable, and pass it to the container
> in the same way as the NED API key.

Note that the container's ouput files will be written as root. To avoid this you
can set the user ID, e.g.:
```docker
docker run \
    -e NED_API_KEY \
    --volume /local/path/to/output/dir:/data \
    --user 1000:1000 \
    ghcr.io/esi-far/emissionfactor-forecast:0.1.0
```
If your user ID is 1000.

## Building the container image

Note that for model training, historical NED and KNMI data is required, but this is removed
from the container image due to licensing restrictions. The required files are;
- NED.nl:
  - wind, zeewind, zon, electriciteitsmix .csv files
  - years 2021, 2022, 2023, 2024
- KNMI:
  - [Historical weather data from De Bilt](https://cdn.knmi.nl/knmi/map/page/klimatologie/gegevens/uurgegevens/uurgeg_260_2021-2030.zip) (as .txt, i.e., `uurgeg_260_2021-2030.txt`)

These NED.nl files are available after registering.

## Local installation, training and prediction

Instead of the containerized model, you can also work in a local environment.

- Download the data from NED.nl, see the previous section for which files you need
- Clone this repository, change working directory into the repository 
- In a Python environment (3.10/3.11) do:

```sh
pip install autogluon.timeseries --extra-index-url https://download.pytorch.org/whl/cpu
pip install -e .[dev]
```

- Set the following environmental variables:
  - `MODEL_PATH` should refer to a directory where the trained model should be stored
  - `TRAINING_DATA` should refer to the directory with the training data .csv files
  - `NED_API_KEY` should be your API key from NED.nl (available after registration)
  - `OPENMETEO_API_KEY` should be your OpenMeteo API key (only for commercial use).
  - `OUTPUT_PATH` should be the path where you want the output .csv files to be written to
- Now you can run `python src/emissionfactor_nl/train_model.py` to train the model
- With `python src/emissionfactor_nl/predict.py` you can generate a forecast based on the currently available forecast data from NED.nl
