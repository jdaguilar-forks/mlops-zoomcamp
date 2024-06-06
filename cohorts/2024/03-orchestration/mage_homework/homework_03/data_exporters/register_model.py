import os
import pickle
import click
import mlflow
import pathlib

from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression


EXPERIMENT_NAME = "mage-homework-03"
RF_PARAMS = ['max_depth', 'n_estimators', 'min_samples_split', 'min_samples_leaf', 'random_state']

mlflow.set_tracking_uri("http://mlflow:5000")
mlflow.set_experiment(EXPERIMENT_NAME)
mlflow.sklearn.autolog()


if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter

@data_exporter
def run_register_model(args):
    dv = args[0]
    lr = args[1]
    
    # Save Model
    mlflow.sklearn.log_model(lr, artifact_path="models")

    pathlib.Path("models").mkdir(exist_ok=True)    
    with open("models/dictvectorizer.b", "wb") as f_out:
        pickle.dump(dv, f_out)
    mlflow.log_artifact("models/dictvectorizer.b", artifact_path="dictvectorizer")