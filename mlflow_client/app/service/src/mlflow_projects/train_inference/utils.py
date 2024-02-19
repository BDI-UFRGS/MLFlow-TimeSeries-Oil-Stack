import os
import sys
sys.path.append('..')

import inspect
import pandas as pd 
import numpy as np
import mlflow
import pytorch_lightning as pl
import xgboost
from darts.utils.utils import series2seq
from compress_pickle import dump, load
from darts.models.forecasting.forecasting_model import (
    ForecastingModel,
    LocalForecastingModel,
)
from darts import TimeSeries
import requests
import json
from requests.auth import HTTPBasicAuth
from decouple import config

cur_dir = os.path.dirname(os.path.realpath(__file__))

class ConfigParser:
    def __init__(self, config_file_path=f'{cur_dir}'):
        import yaml
        with open(f'{config_file_path}/config.yml', "r") as ymlfile:
            self.config = yaml.safe_load(ymlfile)
        with open(f'{config_file_path}/data_types.yml', "r") as ymlfile:    
            self.data_types = yaml.safe_load(ymlfile)
            # self.mlflow_tracking_uri = self.config['mlflow_settings']['mlflow_tracking_uri']

    def read_hyperparameters(self, hyperparams_entrypoint):
        return self.config['hyperparameters'][hyperparams_entrypoint]

    def read_entrypoints(self):
        return self.config['hyperparameters']

    def read_data_types(self):
        return self.data_types['data_types']
    
def split_cv_timeseries(df, n_splits):
    splits = []
    for i in range(1,n_splits+1):
        
        test_size = ((max(df['time_idx'])+1) //(n_splits + 1)) / 2
        train_size = i * (max(df['time_idx'])+1) // (n_splits + 1) + (max(df['time_idx'])+1) % (n_splits + 1)
        
        new_df_train = df.loc[df['time_idx'] < train_size]
        new_df_val = df.loc[(df['time_idx'] >= train_size) & (df['time_idx'] < (train_size+test_size))]
        new_df_test = df.loc[(df['time_idx'] >= (train_size+test_size)) & (df['time_idx'] < (train_size+test_size+test_size))]
                
        splits.append((new_df_train,new_df_val, new_df_test))
                
    return splits

class PrintAccuracyAndLossPL(pl.Callback):    
    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        if not trainer.sanity_checking:
            val_loss = trainer.callback_metrics["val_loss"]
            mlflow.log_metrics({'val_loss': val_loss})
    def on_train_epoch_end(self, trainer, pl_module):
        if not trainer.sanity_checking:
            train_loss = trainer.callback_metrics["train_loss"]
            mlflow.log_metrics({'train_loss': train_loss})

class PrintAccuracyAndLossXGB(xgboost.callback.TrainingCallback):
    def after_training(self, model, epoch, evals_log):
        return False

def load_yaml_as_dict(filepath):
    import yaml
    with open(filepath, 'r') as stream:
        try:
            parsed_yaml = yaml.safe_load(stream)
            return parsed_yaml
        except yaml.YAMLError as exc:
            print(exc)
            return
    
def load_local_pl_model(model_root_dir):
    from darts.models import RNNModel, BlockRNNModel, NBEATSModel, TFTModel, NaiveDrift, NaiveSeasonal, TCNModel, NHiTSModel, TransformerModel
    print("\nLoading local PL model...")
    
    model_info_dict = load_yaml_as_dict(os.path.join(model_root_dir, 'model_info.yml'))

    darts_forecasting_model = model_info_dict["darts_forecasting_model"]

    model = eval(darts_forecasting_model)

    # model_root_dir = model_root_dir.replace('/', os.path.sep)

    print(f"Loading model from local directory:{model_root_dir}")

    best_model = model.load_from_checkpoint(model_root_dir, best=True)

    return best_model

def load_local_pkl_as_object(local_path):
    import pickle
    pkl_object = pickle.load(open(local_path, "rb"))
    return pkl_object

def load_model(model_root_dir):

    # Get model type as tag of model's run
    import mlflow
    print(model_root_dir)

    client = mlflow.tracking.MlflowClient()
    run_id = model_root_dir.split('/')[-1]
    model_run = client.get_run(run_id)
    model_type = model_run.data.tags.get('model_type')
    
    targets = model_run.data.params.get('targets')
    past_cov = model_run.data.params.get('past_cov')
    future_cov = model_run.data.params.get('future_cov')
    interpolate = model_run.data.params.get('interpolate')
    add_func = model_run.data.params.get('add_func')
    
    # Load accordingly
    if model_type == 'pl':
        model = load_local_pl_model(model_root_dir=model_root_dir)
    elif model_type == 'pkl':
        model_uri = os.path.join(model_root_dir, '_model.pkl')
        model = load_local_pkl_as_object(model_uri)
        
    scalers_uri = os.path.join(model_root_dir, '_scalers.pkl')
    scalers = load_local_pkl_as_object(scalers_uri)
                
    return model, scalers, interpolate, add_func, targets, past_cov, future_cov

def conn_load_dataset_local_pkl():
    return load("dataset.gz", compression='gzip')['2018-01-01':'2019-08-30']

def add_time_idx(df):
    df['time_idx'] = np.arange(len(df))
    return df

def temp_add_aditional_features(df, interpolate_method='linear'):
    # exemple for function to add additional features

    return df

def very_big_number_outlier_removal(df):
    df[(df > 1000000)] = np.nan  
    return df
    
def std_dev_12h_outlier_removal(df):
    rolling_mean = df.rolling(window='12h').mean()
    rolling_std = df.std()
    lower_bound = rolling_mean - (1 * rolling_std)
    upper_bound = rolling_mean + (1 * rolling_std)
    df[(df< lower_bound) | (df > upper_bound)] = np.nan  
    df[(df > 1000000)] = np.nan  

    return df

def std_dev_global_outlier_removal(df):
    rolling_mean = df.mean()
    rolling_std = df.std()
    lower_bound = rolling_mean - (2 * rolling_std)
    upper_bound = rolling_mean + (2 * rolling_std)
    df[(df< lower_bound) | (df > upper_bound)] = np.nan  
    df[(df > 1000000)] = np.nan  

    return df
    
def conn_load_dataset_remote(freq='1min', features=[],start_time='2018-10-18T00:00:00', end_time='2018-10-21T00:00:00', except_remove_outliers=[], remove_outliers=very_big_number_outlier_removal, interpolate_method='linear'):
    # implement connector for remote database
    # df = get_database()
    return df

class NaiveLastValueModel(LocalForecastingModel):
    def __init__(self):
        super().__init__()
        self.rolling_window = None

    @property
    def supports_multivariate(self) -> bool:
        return True

    @property
    def min_train_series_length(self):
        return 1

    def __str__(self):
        return f"NaiveLastValue()"

    def fit(self, series: TimeSeries):
        super().fit(series)
        self.rolling_window = series[-1:].values(copy=False)
        return self


    def predict(
        self,
        n: int,
        verbose: bool = False,
        num_samples: int = 1
    ):
        super().predict(n, num_samples)

        predictions_with_observations = np.concatenate(
            (self.rolling_window, np.full(shape=(n, self.rolling_window.shape[1]), fill_value=self.rolling_window[0][0])),
            axis=0,
        )

        return self._build_forecast_series(predictions_with_observations[-n:])