# script to train, evaluate and version a baseline Logistic Regression model
# author: jraposoneto

# Import mlflow
import yaml
import mlflow
import pandas as pd
import numpy as np
import joblib
import mlflow.sklearn
from mlflow import MlflowClient
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score, recall_score, classification_report

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

# train baseline function
def train_baseline():
    print('Starting baseline training script...')
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
    
    # hyperparameters tuned with Optuna on Colab Env
    params = {'C': 3.2436522570149235, 'solver': 'liblinear', 'random_state': 777,
              'class_weight': {0: 1, 1: 3}}
    
    print(50*'=')
    
    # Wrap autoML training with MLflow
    with mlflow.start_run(run_name = f'baseline_logistic_reg'):
        model = LogisticRegression(
            **params
        )
        print('Calculating Cross Validation metrics...')
        cv_results_auc = cross_val_score(model, Xtrain, ytrain.values.reshape(-1,), scoring='roc_auc',
                cv = 10)
        
        cv_results_recall = cross_val_score(model, Xtrain, ytrain.values.reshape(-1,), scoring='recall',
                cv = 10)
        
        model.fit(Xtrain.values, ytrain.values.reshape(-1,))
        preds = model.predict(Xval.values)
        probas = model.predict_proba(Xval.values)
        
        # Set metrics to log
        # log params
        mlflow.log_params(params)
        mlflow.log_metrics({"Mean 10 Fold CV AUC": np.round(np.mean(cv_results_auc),2),
                            "Stdev 10 Fold CV AUC": np.round(np.std(cv_results_auc),2),
                            "Mean 10 Fold CV Recall": np.round(np.mean(cv_results_recall),2),
                            "Stdev 10 Fold CV Recall": np.round(np.std(cv_results_recall),2),
                            "OOS Validation ROC AUC": roc_auc_score(yval.values.reshape(-1,), probas[:,1]),
                            "OOS Validation recall": np.round(recall_score(yval.values.reshape(-1,), preds),2)})
        mlflow.sklearn.log_model(model, artifact_path = 'model')
    joblib.dump(model, path['models']['baseline'])
    mlflow.end_run()
    print('=====FINISH!=====')
    
if __name__ == '__main__':
    train_baseline()
