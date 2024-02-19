FROM python:3.9

# setup packages
RUN apt-get update -y
RUN apt-get install -y python3 python-is-python3 python3-pip
RUN pip install --upgrade pip

# install python requirements before copying the rest of the files
# this way we can cache the requirements and not have to reinstall them
COPY requirements/ /app/requirements/
RUN --mount=type=cache,target=/root/.cache/pip \
        pip install -r /app/requirements/all.txt

# copy local files
COPY . /app

# set work directory
WORKDIR /app

CMD ["uvicorn", "app.main:app", "--host=0.0.0.0", "--port=8000", "--reload"]