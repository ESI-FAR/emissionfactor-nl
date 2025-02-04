FROM python:3.11.11-bookworm@sha256:adb581d8ed80edd03efd4dcad66db115b9ce8de8522b01720b9f3e6146f0884c

LABEL maintainer="Bart Schilperoort <b.schilperoort@esciencecenter.nl>"
LABEL org.opencontainers.image.source="https://github.com/ESI-FAR/emissionfactor-nl"

# uv speeds up install
RUN pip install uv
RUN uv pip install autogluon.timeseries==1.2.0 \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        --system

# Copy the local package and install it
COPY src repo/src
COPY pyproject.toml repo/pyproject.toml
COPY LICENSE repo/LICENSE
COPY README.md repo/README.md
RUN pip install -e repo/

# Copy training data from local directory into container
RUN mkdir training_data
COPY ./data/NED training_data

ENV TRAINING_DATA=training_data
ENV MODEL_PATH=model

RUN python repo/src/emissionfactor_nl/train_model.py

# Remove training data again (licensing issues)
RUN rm training_data -rf

# Make model data dir available for all users running docker non-root
RUN chmod -R 777 ${MODEL_PATH}
# Make more required dirs accesible for non-root users
RUN mkdir -m 777 mpl
ENV MPLCONFIGDIR=mpl
RUN mkdir -m 777 hug_cache
ENV HF_HOME=hug_cache

ENV OUTPUT_PATH=/data

CMD [ "python", "repo/src/emissionfactor_nl/predict.py" ]
