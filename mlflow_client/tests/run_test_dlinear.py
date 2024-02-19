import mlflow
from mlflow.entities import RunStatus
from mlflow.tracking import MlflowClient
from mlflow.tracking.fluent import _get_experiment_id
from mlflow.utils import mlflow_tags
from mlflow.utils.logging_utils import eprint

# Run MLflow project and create a reproducible conda environment
# on a local host

#
#with mlflow.start_run(run_name="RAPIDS-Hyperopt") as active_run:
print(mlflow.get_artifact_uri())

features = {
    "1d": [
    ],
    
    "1h": [
    ],
    
    "10min":[
    ]
}

for interpolate_method in ['linear']:
    for outlier_removal in ['very_big_number_outlier_removal']:
        for freq in ["1d","1h","10min"]:
            for input_chunk_length in [1,6,12,24]:
                for output_chunk_length in [1,6,12,24]:
                    if(input_chunk_length >= output_chunk_length):
                        mlflow.set_experiment(experiment_name=f"DLine test {freq} (1)")
                        train_parameters = {"darts_model": "DLinearModel", 
                                                "num_splits":3, 
                                                "optuna_trials":60,
                                                "freq":freq, 
                                                "input_chunk_length":input_chunk_length,
                                                "output_chunk_length":output_chunk_length,
                                                "hyperparams_entrypoint": "dlinear1", 
                                                "targets": "RATE_OIL_PROD",
                                                "future_cov": "CHOKE",
                                                "scaler":"MinMaxScaler",
                                                "past_cov": ','.join(features[freq]),
                                                "interpolate": interpolate_method, 
                                                "outlier_removal": outlier_removal}
                        
                        submitted_run = mlflow.run("../app/service/src/mlflow_projects/train_inference", entry_point="train", parameters=train_parameters, env_manager='local')
                        print(MlflowClient().get_run(submitted_run.run_id))
