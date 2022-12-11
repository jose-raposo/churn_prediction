# script to train, evaluate and version a H2O autoML model
# author: jraposoneto

# Importing
import yaml
import mlflow
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import mlflow.h2o
from mlflow import MlflowClient
import h2o
from h2o.automl import H2OAutoML, get_leaderboard
import time
from joblib import Memory 


# load yaml main file with parameters
with open("../config/config.yaml", "r") as file:
    try:
        path = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)

# loading preprocessed data
Xtrain = pd.read_parquet(path['preprocessed']['Xtrain'])
ytrain = pd.read_parquet(path['preprocessed']['ytrain'])

Xval = pd.read_parquet(path['preprocessed']['Xval'])
yval = pd.read_parquet(path['preprocessed']['yval'])

train = pd.concat([Xtrain, ytrain['Churn'].map({1:'Yes',0:'No'})], axis = 1)
val = pd.concat([Xval, yval['Churn'].map({1:'Yes',0:'No'})], axis = 1)

del Xtrain
del ytrain
del Xval
del yval

# start local H2O env
h2o.init()

# h2o specific frames
train = h2o.H2OFrame(train)
val = h2o.H2OFrame(val)

# Create a memory object 
mem = Memory(path['cache'])

# train AutoML model
@mem.cache(verbose=0)
def automl():
    start = time.time()
    print(f'Starting baseline training script... Time: {start}')
    print(50*'=')
    # Initialize MLFlow client
    client = MlflowClient()

    # Set up MlFlow experiment
    experiment_name = 'churn-prediction'

    try:
        experiment_id = mlflow.create_experiment(experiment_name)
        experiment = client.get_experiment_by_name(experiment_name)
    except:
        experiment = client.get_experiment_by_name(experiment_name)
        
    mlflow.set_experiment(experiment_name)

    # Print experiment details
    print(f"Name: {experiment_name}")
    print(f"Experiment_id: {experiment.experiment_id}")
    print(f"Artifact Location: {experiment.artifact_location}")
    print(f"Tags: {experiment.tags}")
    print(f"Lifecycle_stage: {experiment.lifecycle_stage}")
    print(f"Tracking uri: {mlflow.get_tracking_uri()}")
    
    with mlflow.start_run(run_name = 'h2o_automl_model'):
        aml = H2OAutoML(
                        max_models=100, # Run AutoML for n base models
                        seed=777, 
                        balance_classes=True, # Our target classes are imbalanced, so we set this to True
                        sort_metric='AUC', # Sort models by logloss (main metric for multi-classification)
                        verbosity='info', # Turn on verbose info
                        nfolds = 10,
                        max_runtime_secs = 1200
                    )
        start = time.time()
        print(f'Starting H2O training script... Time: {start}')
        aml.train(training_frame = train, y = 'Churn')
        end = time.time()
        print(f'Finalizing H2O training script... Time: {end}')
        print(f'Wall Time: {end - start}')
        print(50*'=')
        
        # Set metrics to log
        print('Logging training model metrics of leader model')
        mlflow.log_metric("log_loss", aml.leader.logloss())
        mlflow.log_metric("AUC", aml.leader.auc())
        
        print(80*'=')
        print('Logging best model')
        # Log best model (mlflow.h2o module provides API for logging & loading H2O models)
        mlflow.h2o.log_model(aml.get_best_model(), 
                            artifact_path="best_model"
                            )
        
        model_uri = mlflow.get_artifact_uri("model")
        print(model_uri)

        print('Showing leaderboard')
        print(80*'=')
        # Print and view AutoML Leaderboard
        lb = get_leaderboard(aml, extra_columns='ALL')
        print(lb.head(rows=lb.nrows))
        
        # Get IDs of current experiment run
        exp_id = experiment.experiment_id
        run_id = mlflow.active_run().info.run_id
        
        # Save leaderboard as CSV
        print(80*'=')
        print('Saving leaderboard as csv')
        lb_path = f'/home/jraposoneto/churn_prediction/src/mlruns/{exp_id}/{run_id}/artifacts/leaderboard.csv'
        lb.as_data_frame().to_csv(lb_path, index=False) 
        print(f'Leaderboard saved in {lb_path}')
        
        print(80*'=')
        print('Printing model log events')
        print(aml.event_log)
        print(80*'=')

        print('Learning Curve...')
        aml.leader.learning_curve_plot()
    
        best_model = aml.get_best_model()
        val_perf = best_model.model_performance(val)
        
        print(80*'=')
        print('Model explanations...')
        aml.explain(frame = val, figsize = (8,6))
        print(80*'=')
        print('Validation performance...')
        print(val_perf)
 
    mlflow.end_run()
    print('=====FINISH!=====')
    
if __name__ == '__main__':
    automl()