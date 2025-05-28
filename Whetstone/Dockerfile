FROM --platform=linux/x86_64 python:3.9

# Copy application dependency manifests to the container image.
# Copying this separately prevents re-running pip install on every code change.
COPY requirements.txt ./

# Install production dependencies.
RUN set -ex; \
    apt-get update; \
    apt-get install vim -y; \
    pip install -r requirements.txt

#RUN python3 -m pip install --upgrade https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-0.12.0-py3-none-any.whl

# Copy local code to the container image.
ENV APP_HOME /app
WORKDIR $APP_HOME
COPY project/ project/
WORKDIR $APP_HOME/project

