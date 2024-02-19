# MLFlow Oil Production Time-Series Forecast Stack

This repository contains a microservice stack based on MLFlow for training and executing Oil production time-series forecasting Machine-Learning Models. It is a proposal for a digital twin module.

1. The project tree has a docker-compose.yml for deployment in a *docker swarm*. The architecture consists of:
    * mlflow_client
        * app
            * service
                * src
                    * **forecast.py**: API service that receives user parameters and executes the MLProject *backcast* in *projects/train_inference*
            
            * **main.py**: main application file that executes and keeps all FastAPI endpoints
        * projects
            * train_inference
                * **train.py**: Trains forecasting models 
                * **backcast.py**: Executes model inference
                * **utils.py**: Keeps utility functions used throughout the code 
                * **config.yml**: Contains all adjustable parameters for model training (hyperparameter ranges, size of windows, num of trials, etc)
                * **data_types.yml**: Contains a dictionary with a datatype of each hyperparameter
                * **registered_models.yml**: Contains a list of registered models for each combination of lookback, horizon and sampling frequency
        * requirements
        * tests
    * mlflow_tracking
    * mysql
    * minio
