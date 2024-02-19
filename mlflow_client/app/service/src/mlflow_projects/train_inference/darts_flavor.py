import os
from utils import load_model
from urllib3.exceptions import InsecureRequestWarning
from urllib3 import disable_warnings
disable_warnings(InsecureRequestWarning)

class _MLflowPLDartsModelWrapper:
    def __init__(self, python_model, scalers, interpolate, add_func, targets, past_cov, future_cov):
        self.python_model = python_model
        self.scalers = scalers
        self.interpolate = interpolate
        self.add_func=add_func
        self.targets=targets
        self.past_cov=past_cov
        self.future_cov=future_cov

    def predict(self, model_input):
        model_input['series'] = self.scalers[0].transform(model_input['series'])
        model_input['past_covariates'] = self.scalers[1].transform(model_input['past_covariates'])
        model_input['future_covariates'] = self.scalers[2].transform(model_input['future_covariates'])
        
        predictions = self.python_model.historical_forecasts(**model_input)
        predictions = self.scalers[0].inverse_transform(predictions[0])
        
        return predictions

def _load_pyfunc(model_folder):
    model_folder = model_folder.replace('/', os.path.sep)
    model_folder = model_folder.replace('\\', os.path.sep)
    
    model, scalers, interpolate, add_func, targets, past_cov, future_cov = load_model(model_root_dir=model_folder)

    return _MLflowPLDartsModelWrapper(model, scalers, interpolate, add_func, targets, past_cov, future_cov)