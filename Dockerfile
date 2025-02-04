FROM python:3.11.11-bookworm@sha256:adb581d8ed80edd03efd4dcad66db115b9ce8de8522b01720b9f3e6146f0884c

RUN pip install autogluon.timeseries==1.2.0 --extra-index-url https://download.pytorch.org/whl/cpu

RUN mkdir training_data
COPY ./data/NED training_data
COPY ./src/emissionfactor-nl/train_model.py train_model.py
COPY ./src/emissionfactor-nl/read_ned.py read_ned.py

ENV TRAINING_DATA=training_data
ENV MODEL_PATH=model

RUN python train_model.py

# Remove training data again (licensing issues)
RUN rm training_data -rf

# Make model data dir available for all users running docker non-root
RUN chmod -R 777 ${MODEL_PATH}
# Make more required dirs accesible for non-root users
RUN mkdir -m 777 mpl
ENV MPLCONFIGDIR=mpl
RUN mkdir -m 777 hug_cache
ENV HF_HOME=hug_cache

COPY ./src/emissionfactor-nl/retrieve_ned.py retrieve_ned.py
COPY ./src/emissionfactor-nl/predict.py predict.py
ENV OUTPUT_PATH=/data

CMD [ "python", "predict.py" ]
