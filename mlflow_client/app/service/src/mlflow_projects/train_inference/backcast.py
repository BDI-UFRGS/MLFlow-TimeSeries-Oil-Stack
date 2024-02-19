import sys
import os
sys.path.append('..')

import mlflow
import tempfile
from utils import conn_load_dataset_local_pkl, add_time_idx, temp_add_aditional_features, very_big_number_outlier_removal

from darts import TimeSeries
import numpy as np
import pandas as pd

from datetime import datetime, timedelta
import random

import tempfile

def MLflowDartsModelPredict(pyfunc_model_folder, forecast_horizon=6, forecast_date='2018-08-08', freq='1h', choke="100"):
    forecast_date = datetime.strptime(forecast_date, '%Y-%m-%d %H:%M:%S')
    forecast_date = forecast_date + timedelta(hours=3)
    forecast_date = forecast_date.strftime('%Y-%m-%d %H:%M:%S')

    predictions = None
    real_inference = True
    
    with mlflow.start_run(run_name='inference') as mlrun:
        loaded_model = mlflow.pyfunc.load_model(pyfunc_model_folder)

        client = mlflow.tracking.MlflowClient()
        model_run = client.get_run(loaded_model.metadata.run_id)
        
        input_chunk_length = model_run.data.tags.get('input_chunk_length')
        add_func = model_run.data.tags.get('add_func')
        interpolate_method = model_run.data.tags.get('interpolate')

        targets= model_run.data.params.get('darts_targets').split(",")
        future_cov= model_run.data.params.get('darts_future_cov').split(",")
        past_cov= model_run.data.params.get('darts_past_cov').split(",")
        output_chunk_length= model_run.data.params.get('constructor_output_chunk_length')

        df = conn_load_dataset_local_pkl()
        df = eval(add_func)(df.resample(freq).mean())
        df = add_time_idx(df).interpolate(method=interpolate_method).fillna(method="bfill")

        output_chunk_length = int(output_chunk_length)
        forecast_horizon = int(forecast_horizon)
        input_chunk_length = int(input_chunk_length)

        new_df_past = TimeSeries.from_dataframe(df[df.index <= forecast_date].tail(input_chunk_length*2), value_cols=targets+future_cov+past_cov,fill_missing_dates=True)
        new_df_future = TimeSeries.from_dataframe(df[df.index > forecast_date].head(output_chunk_length*2), value_cols=targets+future_cov+past_cov,fill_missing_dates=True)

        train_target = new_df_past[targets].append(new_df_future[targets])
        train_past_cov = new_df_past[past_cov].append(new_df_future[past_cov])
        
        if(choke == "Historical"):
            train_future_cov= new_df_past[future_cov].append(new_df_future[future_cov])  
        else:
            train_future_cov= new_df_past[future_cov].append(new_df_future[future_cov].map(lambda x: np.full((len(new_df_future), 1, 1), int(choke))))
        
        train_future_cov = train_future_cov.astype(np.float32)

        input = {
            "series": train_target,
            "past_covariates": train_past_cov,
            "future_covariates": train_future_cov,
            "start": len(new_df_past),
            "forecast_horizon": output_chunk_length,
            "retrain": False,
            "last_points_only": False,
            "start_format":"position",
        }

        if(real_inference):
            historical_forecasts = loaded_model.predict(input).pd_dataframe().tz_localize('UTC')
        else:
            def random_forecaster(x):
                random.seed(output_chunk_length+x.values[0])
                return x+random.gauss(0, 10)

            historical_forecasts = new_df_future[targets][:output_chunk_length].pd_dataframe().tz_localize('UTC').apply(lambda x: random_forecaster(x), axis = 1)

        historical_forecasts.index = historical_forecasts.index - pd.Timedelta(hours=3)
        predictions = historical_forecasts[:forecast_horizon]

        infertmpdir = tempfile.mkdtemp()
        predictions.to_csv(os.path.join(infertmpdir, 'predictions.csv'))
        mlflow.log_artifacts(infertmpdir)

    return predictions

if __name__ == '__main__':
    print("\n INFERENCE")
    MLflowDartsModelPredict(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])