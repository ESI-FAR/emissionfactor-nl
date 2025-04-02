# Change Log

All notable changes to this project will be documented in this file.
This project adheres to [Semantic Versioning](http://semver.org/).
Formatted as described on [https://keepachangelog.com](https://keepachangelog.com/en/1.0.0/).

## Unreleased

## [0.2.0] (2025-04-02)

**Added:**

- Air temperature data is used for an improved model
  - Historic data can be retrieved from the KNMI
  - Forecasts are retrieved via the OpenMeteo API
- Total green production is used as input to the model, instead of separate components

## [0.1.0] (2025-02-04)

With this workflow you can make forecasts of the emission factor (kgCO2eq/kWh)
of the electricity grid of the Netherlands.

The forecast is based on data sourced from the [Nationaal Energie Dashboard](https://ned.nl/).

Included in this repository are both the code to reproduce the model, as well as a 
(containerized) workflow to produce forecasts.

0.1.0 marks the first release of this model, compatibility between this version and
newer versions is not guaranteed.
