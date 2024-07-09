import pandas as pd 
import sktime 

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor

from sktime.forecasting.base import BaseForecaster


class LocalYAFSUModels(BaseForecaster):