import pickle
import os
import mlflow
import logging
from mlflow.tracking import MlflowClient
import pandas as pd
import json
import yaml
from fastapi import FastAPI, HTTPException

def forecast_darts(origin_timestamp="2018-08-08", forecast_length=6, frequency="1h", choke=100, target='RATE_OIL_PROD'):
    with open(f'app/service/src/mlflow_projects/train_inference/registered_models.yml', "r") as ymlfile:
        registered_models = yaml.safe_load(ymlfile)

    try:
        available_output_chunk_lengths = list(map(int, registered_models["registered_models"][target][frequency].keys()))
        largest_closest_output_length = [y for y in available_output_chunk_lengths if y >= int(forecast_length)]
    
        registered_model = registered_models["registered_models"][target][frequency][f"{largest_closest_output_length[0]}"]
    except:
        raise HTTPException(status_code=404, detail="No model found for this forecast length or frequency")
    
    params= {
        "pyfunc_model_folder":registered_model,
        "forecast_date":origin_timestamp,
        "forecast_horizon":forecast_length,
        "freq":frequency,
        "choke": choke
    }
    
    submitted_run = mlflow.run("app/service/src/mlflow_projects/train_inference", entry_point="backcast", parameters=params, env_manager='local')

    config_json = mlflow.artifacts.download_artifacts(MlflowClient().get_run(submitted_run.run_id).info.artifact_uri + "/predictions.csv")

    df = pd.read_csv(config_json, index_col='timestamp')
    df.index = df.index.rename("x")
    df.columns = ['y'] + ['y_' + str(i) for i in range(2, len(df.columns) + 1)]
    df['x'] = df.index
    
    df.index = pd.to_datetime(df.index)
    df.index = df.index.map(lambda x: x.isoformat())
    
    json_str = json.dumps(df.to_json(orient="records"))

    return json_str