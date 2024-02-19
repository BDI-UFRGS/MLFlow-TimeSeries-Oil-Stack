FROM python:3.10-slim-buster
# Install python packages
COPY requirements.txt /tmp
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r /tmp/requirements.txt

COPY . /app

WORKDIR /app