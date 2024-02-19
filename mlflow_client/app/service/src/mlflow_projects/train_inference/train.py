from random import randint
from mlflow.pyfunc import PythonModel
import mlflow
import pandas as pd
from darts import TimeSeries
from sklearn.preprocessing import MinMaxScaler
from darts.dataprocessing.transformers import Scaler
from darts.models import ExponentialSmoothing, NaiveMean, TFTModel, RNNModel, XGBModel, RegressionModel, DLinearModel, AutoARIMA, RandomForest
import logging
import tempfile
import pickle
import os
import shutil
import click
from utils import ConfigParser, conn_load_dataset_local_pkl, add_time_idx, split_cv_timeseries, PrintAccuracyAndLossPL, PrintAccuracyAndLossXGB, std_dev_12h_outlier_removal, std_dev_global_outlier_removal, backtest_unscaled, NaiveLastValueModel, temp_add_aditional_features
import torch
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from darts.utils.model_selection import train_test_split
import optuna
from optuna.integration import PyTorchLightningPruningCallback
from darts.metrics import rmse
import numpy as np
import logging
import pytorch_lightning as pl
import ast
import yaml

logging.getLogger("pytorch_lightning.utilities.rank_zero").setLevel(logging.WARNING)
logging.getLogger("pytorch_lightning.accelerators.cuda").setLevel(logging.WARNING)

mlflow_model_root_dir = "pyfunc_model"


@click.command()
@click.option("--start_date",
        type=str,
        default="2018-01-01T00:00:00",
        help="Start date of training data")

@click.option("--end_date",
        type=str,
        default="2019-08-31T00:00:00",
        help="End date of training data")

@click.option("--darts_model",
              type=click.Choice(
                  ['RegressionModel',
                    'TFTModel',
                    'RNNModel',
                     'XGBModel',
                     'NaiveLastValueModel',
                   'DLinearModel',
                   'AutoARIMA',
                   'RandomForest'
                  ]),
              multiple=False,
              default='NaiveMean',
              help="The Darts model to be trained"
              )
@click.option("--num_splits",
        type=int,
        default="3",
        help="Number of splits for nested cross-validation")

@click.option("--optuna_trials",
        type=int,
        default="50",
        help="Number of trials used by optuna for hyperparameter search")

@click.option("--freq",
        type=str,
        default="10min",
        help="Sampling frequency of observations")

@click.option("--input_chunk_length",
        type=int,
        default=1,
        help="Size of lookback")

@click.option("--output_chunk_length",
        type=int,
        default=6,
        help="Size of horizon")

@click.option("--hyperparams_entrypoint", "-h",
              type=str,
              default='naive1',
              help="The entry point in config.yml containing the desired hyperparameters for the selected model"
              )
@click.option("--device",
              type=click.Choice(
                  ['gpu',
                   'cpu']),
              multiple=False,
              default='gpu',
              help="""If to use 'cpu' or 'gpu' for training"""
              )
@click.option("--num_workers",
        type=str,
        default="4",
        help="Number of threads used for training")

@click.option("--targets",
        callback=lambda _,__,x: x.split(',') if x else [],
        default="",
        help="List of target features")

@click.option("--future_cov",
        callback=lambda _,__,x: x.split(',') if x else [],
        default="",
        help="List of future covariates")

@click.option("--past_cov",
        callback=lambda _,__,x: x.split(',') if x else [],
        default="",
        help="List of past covariates")

@click.option('--interpolate', 
              type=click.Choice(
                  ['linear',
                   'ffill']),
              multiple=False,
              default='linear',
              help="The filling strategy for gaps in timeseries, either linear_interpol or ffill (forward fill)")

@click.option('--outlier_removal', 
              type=click.Choice(
                  ['std_dev_12h',
                  'std_dev_global',
                  'very_big_number_outlier_removal']),
              multiple=False,
              default='very_big_number_outlier_removal',
              help="The strategy for removing outliers, either std_dev_12h or std_dev_global")

@click.option('--add_func', 
              type=click.Choice(
                  ['temp_add_aditional_features']),
              multiple=False,
              default='temp_add_aditional_features',
              help="The strategy for adding extra features")

@click.option("--scaler",
              type=click.Choice(
                  ['MinMaxScaler']),
              multiple=False,
              default='MinMaxScaler',
              help="The feature scaler to use"
              )

def train(start_date, end_date, darts_model, num_splits, optuna_trials, freq, input_chunk_length, output_chunk_length, hyperparams_entrypoint, device, num_workers, targets, future_cov, past_cov, interpolate, outlier_removal, add_func, scaler):
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    current_dir = os.getcwd()

    with mlflow.start_run() as mlrun:
        num_workers = int(num_workers)
        torch.set_num_threads(num_workers)
    
        ######################
        ## Set device
        if device == 'gpu' and torch.cuda.is_available():
            device = 'gpu'
            print("\nGPU is available")
        else:
            device = 'cpu'
            print("\nGPU is available")
    
        ######################
        ## Compile tunable and non-tunable hyperparameters
        hyperparameters = ConfigParser().read_hyperparameters(hyperparams_entrypoint)
        if(hyperparameters is None):
            hyperparameters = {}
        hyperparameters["output_chunk_length"] = output_chunk_length

        if darts_model in ['TFTModel', 'RNNModel','DLinearModel']:
            hyperparameters['input_chunk_length'] = input_chunk_length
            model_type = "pl"
        elif darts_model in ['XGBModel','RegressionModel', 'RandomForest']:
            hyperparameters['lags'] = input_chunk_length
            hyperparameters['lags_past_covariates'] = input_chunk_length
            hyperparameters['lags_future_covariates'] = (input_chunk_length, hyperparameters["output_chunk_length"])
            model_type = "pkl"
        else:
            del hyperparameters["output_chunk_length"]
            model_type = "pkl"
        
        data_types = ConfigParser().read_data_types()
            
        ######################
        # Load dataset (pkl file or remote)
        df = conn_load_dataset_local_pkl()
        df = df.interpolate(method='linear').fillna(method="bfill").resample(freq).mean()
        #df = conn_load_dataset_remote(freq='10min', tags=[],start_time=start_date, end_time=end_date, except_remove_outliers=[], remove_outliers=eval(outlier_removal), interpolate_method=interpolate)
        #df = eval(add_func)(df.resample(freq).mean())
        #df = add_time_idx(df).interpolate(method='linear').fillna(method="bfill")
        
        ## create splits
        splits = split_cv_timeseries(df, num_splits)
    
        #splits
    
        last_split_id = ""
        
        for i, split in enumerate(splits):
            print(f'\nSplit {i}')

            print(f'\nTrain start')
            print(split[0].iloc[0].name)
            print(f'\nTrain end')
            print(split[0].iloc[-1].name)

            print(f'\nVal start')
            print(split[1].iloc[0].name)
            print(f'\nVal end')
            print(split[1].iloc[-1].name)

            print(f'\nTest start')
            print(split[2].iloc[0].name)
            print(f'\nTest end')
            print(split[2].iloc[-1].name)
            
            with mlflow.start_run(nested=True) as mlrun_split:
                new_df_train,new_df_val, new_df_test = split
                
                new_df_train = TimeSeries.from_dataframe(new_df_train, value_cols=targets+future_cov+past_cov)
                new_df_val = TimeSeries.from_dataframe(new_df_val, value_cols=targets+future_cov+past_cov)
                new_df_test = TimeSeries.from_dataframe(new_df_test, value_cols=targets+future_cov+past_cov)
            
                train_target, val_target, test_target = new_df_train[targets], new_df_val[targets], new_df_test[targets]
                train_future_cov, val_future_cov, test_future_cov = new_df_train[future_cov], new_df_val[future_cov], new_df_test[future_cov]

                if(len(past_cov)==0):
                    train_past_cov, val_past_cov, test_past_cov = None, None, None
                else:
                    train_past_cov, val_past_cov, test_past_cov = new_df_train[past_cov], new_df_val[past_cov], new_df_test[past_cov]

                early_stopper = EarlyStopping("val_loss", min_delta=.01, patience=3, verbose=True)
                callbacks = [early_stopper, PrintAccuracyAndLossPL()]
                
                pl_trainer_kwargs = {"callbacks": callbacks,
                                     "accelerator": 'auto',
                                    #  "gpus": 1,
                                    #  "auto_select_gpus": True,
                                     "log_every_n_steps": 1}
                
                ######################
                # Optimize hyper and fit model
            
                print(f'\nTraining {darts_model} ...')

                study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner())
                preds = study.optimize(lambda trial: trainer_objective(trial, mlrun_split, darts_model, hyperparameters, train_target, train_past_cov, train_future_cov, val_target, val_past_cov, val_future_cov, test_target, test_past_cov, test_future_cov, model_type, scaler, output_chunk_length), n_trials=optuna_trials, callbacks=[print_callback])
                
                best_child_run_id = study.best_trial.user_attrs["run_id"]
    
                best_run = mlflow.get_run(best_child_run_id)
    
                best_params = best_run.data.params
                constructor_params, fit_params = {},{}

                constructor_params = {
                    key.replace('constructor_', ''): eval(data_types[key.replace('constructor_', '')])(value)
                    for key, value in best_params.items()
                    if key.startswith('constructor_')
                }
                
                fit_params = {
                    key.replace('fit_', ''): eval(data_types[key.replace('fit_', '')])(value)
                    for key, value in best_params.items()
                    if key.startswith('fit_')
                }
                
                #for key, value in best_params.items():
                #    constructor_params[key] = eval(data_types[key])(value)
                
                if model_type == "pl":
                    model = eval(darts_model)(
                        log_tensorboard=False,
                        pl_trainer_kwargs=pl_trainer_kwargs,
                        model_name=mlrun_split.info.run_id,
                        force_reset=True,
                        save_checkpoints=True,
                        **constructor_params
                    )
                    
                    logs_path = f"./darts_logs/{mlrun_split.info.run_id}"
                elif model_type == "pkl":
                    model = eval(darts_model)(
                        **constructor_params
                    )

                ###########
                ## Scale vars
                scaler_series = eval(scaler)()
                transformer_series = Scaler(scaler_series)
        
                scaler_past_cov = eval(scaler)()
                transformer_past_cov = Scaler(scaler_past_cov)
        
                scaler_future_cov = eval(scaler)()
                transformer_future_cov = Scaler(scaler_future_cov)

                retrain=False
                
                if darts_model in ['RegressionModel', 'RandomForest']:
                    transformer_series.fit(train_target.append(val_target))
                    transformer_past_cov.fit(train_past_cov.append(val_past_cov))
                    transformer_future_cov.fit(train_future_cov.append(val_future_cov))
                    
                    model.fit(
                          series=transformer_series.transform(train_target.append(val_target)), 
                          past_covariates=transformer_past_cov.transform(train_past_cov.append(val_past_cov)),
                          future_covariates=transformer_future_cov.transform(train_future_cov.append(val_future_cov)),
                          #val_series=val_target,
                          #val_past_covariates=val_past_cov,
                          #val_future_covariates=val_future_cov,
                          **fit_params
                    )
                elif darts_model in ['NaiveLastValueModel']:
                    transformer_series.fit(train_target)
                    transformer_past_cov.fit(train_past_cov)
                    transformer_future_cov.fit(train_future_cov)

                    model.fit(
                      series=transformer_series.transform(train_target.append(val_target)), 
                      **fit_params
                    )

                    retrain=True
                elif darts_model in ['RNNModel']:
                    transformer_series.fit(train_target)
                    transformer_future_cov.fit(train_future_cov)
                    
                    model.fit(
                          series=transformer_series.transform(train_target), 
                          future_covariates=transformer_future_cov.transform(train_future_cov),
                          val_series=transformer_series.transform(val_target),
                          val_future_covariates=transformer_future_cov.transform(val_future_cov),
                          **fit_params
                    )
                elif darts_model in ['AutoARIMA']:
                    transformer_series.fit(train_target)
                    transformer_future_cov.fit(train_future_cov)
                    
                    model.fit(
                          series=transformer_series.transform(train_target), 
                          future_covariates=transformer_future_cov.transform(train_future_cov),
                          **fit_params
                    )
                    retrain=True
                else:
                    transformer_series.fit(train_target)
                    transformer_past_cov.fit(train_past_cov)
                    transformer_future_cov.fit(train_future_cov)
                    
                    model.fit(
                          series=transformer_series.transform(train_target), 
                          past_covariates=transformer_past_cov.transform(train_past_cov),
                          future_covariates=transformer_future_cov.transform(train_future_cov),
                          val_series=transformer_series.transform(val_target),
                          val_past_covariates=transformer_past_cov.transform(val_past_cov),
                          val_future_covariates=transformer_future_cov.transform(val_future_cov),
                          **fit_params
                    )

                if model_type == "pl":
                    model = eval(darts_model).load_from_checkpoint(str(mlrun_split.info.run_id), best=True)
                elif model_type == 'pkl':
                    pass

                historical_forecasts = model.historical_forecasts(
                    transformer_series.transform(train_target.append(val_target).append(test_target)),
                    past_covariates= None if train_past_cov==None else transformer_past_cov.transform(train_past_cov.append(val_past_cov).append(test_past_cov)),
                    future_covariates=transformer_future_cov.transform(train_future_cov.append(val_future_cov).append(test_future_cov)),
                    start=len(train_target)+len(val_target),
                    forecast_horizon=output_chunk_length,
                    retrain=retrain,
                    last_points_only=False,
                )

                backtest_rmse = model.backtest(train_target.append(val_target).append(test_target), 
                                       historical_forecasts=list(map(lambda x: transformer_series.inverse_transform(x), historical_forecasts)),
                                       retrain=retrain,
                                       last_points_only=False,
                                       metric=rmse)
                
                ######################
                # Log metrics
                mlflow.log_metrics({'test_rmse':backtest_rmse})
                
                ######################
                # Log hyperparameters
                constructor_params = {f"constructor_{key}": value for key, value in constructor_params.items()}
                fit_params = {f"fit_{key}": value for key, value in fit_params.items()}

                darts_params = {"darts_targets": ','.join(targets),
                               "darts_future_cov": ','.join(future_cov),
                               "darts_past_cov": ','.join(past_cov)}
                
                mlflow.log_params(constructor_params|fit_params|darts_params)
                
                ######################
                # Log Split
                if model_type == "pl":
                    logs_path_new = logs_path
                elif model_type == 'pkl':
                    print('\nStoring the model as pkl to MLflow...')
                    logging.info('\nStoring the model as pkl to MLflow...')
                    forest_dir = tempfile.mkdtemp()
        
                    pickle.dump(model, open(
                        f"{forest_dir}/_model.pkl", "wb"))
        
                    logs_path = forest_dir
        
                    logs_path_new = logs_path.replace(forest_dir.split('/')[-1], mlrun_split.info.run_id)
                    os.rename(logs_path, logs_path_new)

                pickle.dump((transformer_series,transformer_past_cov,transformer_future_cov), open(f"{logs_path_new}/_scalers.pkl", "wb"))
                
                model_info_dict = {
                    "darts_forecasting_model":  model.__class__.__name__,
                    "run_id": mlrun_split.info.run_id,
                    "frequency": freq,
                    "targets": targets,
                    "past_cov": past_cov,
                    "future_cov": future_cov,
                    "interpolate":interpolate,
                    "outlier_removal":outlier_removal,
                    "scaler":scaler,
                    "train_start":str(split[0].iloc[0].name),
                    "train_end":str(split[0].iloc[-1].name),
                    "val_start":str(split[1].iloc[0].name),
                    "val_end":str(split[1].iloc[-1].name),
                    "test_start":str(split[2].iloc[0].name),
                    "test_end":str(split[2].iloc[-1].name),
                    "add_func": add_func,
                    "input_chunk_length": input_chunk_length
                    }
                with open('model_info.yml', mode='w') as outfile:
                    yaml.dump(
                        model_info_dict,
                        outfile,
                        default_flow_style=False)

                shutil.move('model_info.yml', logs_path_new)
            
                ######################
                # Log artifacts
                
                mlflow.pyfunc.log_model(mlflow_model_root_dir,
                                       loader_module="darts_flavor",
                                       data_path=logs_path_new,
                                       code_path=['./darts_flavor.py',
                                                  #'./utils.py',
                                                 ]
                                       #conda_env=mlflow_serve_conda_env
                                        )
    
                shutil.rmtree(logs_path_new)

                print("\nArtifacts uploaded.")
                logging.info("\nArtifacts uploaded.")
    
                ######################
                # Log tags
                mlflow.set_tag("model_type", model_type)
                mlflow.set_tag("darts_forecasting_model", model.__class__.__name__)
                mlflow.set_tag("frequency", freq)
                mlflow.set_tag("interpolate", interpolate)
                mlflow.set_tag("outlier_removal", outlier_removal)
                mlflow.set_tag("add_func", add_func)
                mlflow.set_tag("scaler", scaler)
                mlflow.set_tag("input_chunk_length", input_chunk_length)
    
                if i==num_splits-1:
                    last_split_id = mlrun_split.info.run_id

        mlflow.set_tag("last_split_id",last_split_id)

        # Calculate the mean of each metric
        child_runs = mlflow.search_runs(filter_string=f"tags.mlflow.parentRunId = '{mlrun.info.run_id}'", output_format='list')
        metric_means = dict.fromkeys(child_runs[0].data.metrics, 0)
        
        #for child_run in child_runs:
        for metric_name, value in child_runs[0].data.metrics.items():
            for child_run in child_runs:
                metric_means[metric_name] += child_run.data.metrics[metric_name]
            metric_means[metric_name] /= num_splits

        mlflow.log_metrics(metric_means)

def trainer_objective(trial, mlrun, darts_model, hyperparameters, train_target, train_past_cov, train_future_cov, val_target, val_past_cov, val_future_cov, test_target, test_past_cov, test_future_cov, model_type, scaler, output_chunk_length):
    with mlflow.start_run(nested=True) as trial_run:
        constructor_params, fit_params = {}, {}

        ## constructor params
        for param, value in hyperparameters.items():
            if type(value) == list and value and value[0] == "range":
                if type(value[1]) == int:
                    constructor_params[param] = trial.suggest_int(param, value[1], value[2], value[3])
                else:
                    constructor_params[param] = trial.suggest_float(param, value[1], value[2], step=value[3])
            elif type(value) == list and value and value[0] == "list":
                constructor_params[param] = trial.suggest_categorical(param, value[1:])
            elif type(value) == list and value and value[0] == "equal":
                continue
            else:
                constructor_params[param] = value
        for param, value in hyperparameters.items():
            if type(value) == list and value and value[0] == "equal":
                constructor_params[param] = constructor_params[value[1]]

        if darts_model in ['TFTModel', 'RNNModel','DLinearModel']:
            if 'learning_rate' in constructor_params:
                constructor_params['optimizer_kwargs'] = {"lr": constructor_params['learning_rate']}
                del constructor_params['learning_rate']

            pruner = PyTorchLightningPruningCallback(trial, monitor="val_loss")
            early_stopper = EarlyStopping("val_loss", min_delta=.001, patience=5, verbose=True)
            callbacks = [pruner, early_stopper, PrintAccuracyAndLossPL()]

            pl_trainer_kwargs = {"callbacks": callbacks,
                             "accelerator": 'auto',
                            #  "gpus": 1,
                            #  "auto_select_gpus": True,
                             "log_every_n_steps": 2}
        
            model = eval(darts_model)(
                log_tensorboard=False,
                pl_trainer_kwargs=pl_trainer_kwargs,
                model_name=trial_run.info.run_id,
                force_reset=True,
                save_checkpoints=True,
                **constructor_params
            )
            
            model_type = "pl"
            logs_path = f"./darts_logs/{trial_run.info.run_id}"
        elif darts_model in ['XGBModel','RegressionModel', 'NaiveLastValueModel',  'AutoARIMA', 'RandomForest']: 
            callbacks = []

            mlflow.xgboost.autolog(log_model_signatures=False, log_models=False, log_datasets=False)
                
            model = eval(darts_model)(
                **constructor_params
            )
            
            model_type = "pkl"

        ###########
        ## Scale vars
        scaler_series = eval(scaler)()
        print(scaler_series)
        transformer_series = Scaler(scaler_series)

        scaler_past_cov = eval(scaler)()
        transformer_past_cov = Scaler(scaler_past_cov)

        scaler_future_cov = eval(scaler)()
        transformer_future_cov = Scaler(scaler_future_cov)

        retrain=False
        
        if darts_model in ['RegressionModel', 'RandomForest']:
            transformer_series.fit(train_target.append(val_target))
            transformer_past_cov.fit(train_past_cov.append(val_past_cov))
            transformer_future_cov.fit(train_future_cov.append(val_future_cov))
            
            model.fit(
                  series=transformer_series.transform(train_target.append(val_target)), 
                  past_covariates=transformer_past_cov.transform(train_past_cov.append(val_past_cov)),
                  future_covariates=transformer_future_cov.transform(train_future_cov.append(val_future_cov)),
                  #val_series=val_target,
                  #val_past_covariates=val_past_cov,
                  #val_future_covariates=val_future_cov,
                  **fit_params
            )
        elif darts_model in ['NaiveLastValueModel']:
            transformer_series.fit(train_target)
            transformer_past_cov.fit(train_past_cov)
            transformer_future_cov.fit(train_future_cov)
            
            model.fit(
                  series=transformer_series.transform(train_target.append(val_target)), 
                  **fit_params
            )

            retrain=True
            
            pass
        elif darts_model in ['RNNModel',]:
            transformer_series.fit(train_target)
            transformer_future_cov.fit(train_future_cov)
            
            model.fit(
                  series=transformer_series.transform(train_target), 
                  future_covariates=transformer_future_cov.transform(train_future_cov),
                  val_series=transformer_series.transform(val_target),
                  val_future_covariates=transformer_future_cov.transform(val_future_cov),
                  **fit_params
            )
        elif darts_model in ['AutoARIMA']:
            transformer_series.fit(train_target)
            transformer_future_cov.fit(train_future_cov)
            
            model.fit(
                  series=transformer_series.transform(train_target), 
                  future_covariates=transformer_future_cov.transform(train_future_cov),
                  **fit_params
            )
            retrain=True
        else:
            transformer_series.fit(train_target)
            transformer_past_cov.fit(train_past_cov)
            transformer_future_cov.fit(train_future_cov)
            
            model.fit(
                  series=transformer_series.transform(train_target), 
                  past_covariates=transformer_past_cov.transform(train_past_cov),
                  future_covariates=transformer_future_cov.transform(train_future_cov),
                  val_series=transformer_series.transform(val_target),
                  val_past_covariates=transformer_past_cov.transform(val_past_cov),
                  val_future_covariates=transformer_future_cov.transform(val_future_cov),
                  **fit_params
            )

        if model_type == "pl":
            model = eval(darts_model).load_from_checkpoint(str(trial_run.info.run_id), best=True)
        elif model_type == 'pkl':
            pass

        historical_forecasts = model.historical_forecasts(
            transformer_series.transform(train_target.append(val_target).append(test_target)),
            past_covariates= None if train_past_cov==None else transformer_past_cov.transform(train_past_cov.append(val_past_cov).append(test_past_cov)),
            future_covariates=transformer_future_cov.transform(train_future_cov.append(val_future_cov).append(test_future_cov)),
            start=len(train_target)+len(val_target),
            forecast_horizon=output_chunk_length,
            retrain=retrain,
            last_points_only=False,
        )
        
        backtest_rmse = model.backtest(train_target.append(val_target).append(test_target), 
                                       historical_forecasts=list(map(lambda x: transformer_series.inverse_transform(x), historical_forecasts)),
                                       retrain=retrain,
                                       last_points_only=False,
                                       metric=rmse)
        
        trial.set_user_attr("run_id", trial_run.info.run_id)
        
        ######################
        # Log FrozenTrial
        if model_type == "pl":
            logs_path_new = logs_path
        elif model_type == 'pkl':
            print('\nStoring the model as pkl to MLflow...')
            logging.info('\nStoring the model as pkl to MLflow...')
            forest_dir = tempfile.mkdtemp()

            #pickle.dump(model, open(f"{forest_dir}/_model.pkl", "wb"))

            logs_path = forest_dir

            logs_path_new = logs_path.replace(forest_dir.split('/')[-1], trial_run.info.run_id)
            os.rename(logs_path, logs_path_new)

        #pickle.dump((transformer_series,transformer_past_cov,transformer_future_cov), open(f"{logs_path_new}/_scalers.pkl", "wb"))

        
        mlflow.pyfunc.log_model(mlflow_model_root_dir,
                                loader_module="darts_flavor",
                                #data_path=f"C:\\Users\\Marco\\petwin-service-darts\\mlflow_client\\projects\\test\\darts_logs\\{str(trial._trial_id)}",
                                data_path=logs_path_new,
                                code_path=['./utils.py','./darts_flavor.py']
                               )
        shutil.rmtree(logs_path_new)
        ######################
        # Log hyperparameters
        
        constructor_params = {f"constructor_{key}": value for key, value in constructor_params.items()}
        fit_params = {f"fit_{key}": value for key, value in fit_params.items()}
        
        mlflow.log_params(constructor_params|fit_params)

        ######################
        # Log tags
        mlflow.set_tag("model_type", model_type)
        mlflow.set_tag("darts_forecasting_model", model.__class__.__name__)

        ######################
        # Log metrics
        mlflow.log_metrics({'test_rmse':backtest_rmse})
    
        return backtest_rmse if backtest_rmse != np.nan else float("inf")

def print_callback(study, trial):
    print(f"Current value: {trial.value}, Current params: {trial.params}")
    print(f"Best value: {study.best_value}, Best params: {study.best_trial.params}")

if __name__ == '__main__':
    print("\n TRAINING")
    print("Tracking uri: {}".format(mlflow.get_tracking_uri()))
    train()

