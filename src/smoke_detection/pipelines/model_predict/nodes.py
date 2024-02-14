"""
This is a boilerplate pipeline
generated using Kedro 0.18.8
"""

import logging
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd


from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder , LabelEncoder
import shap 
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import sklearn
import mlflow

def model_predict_dt(model, X_test: pd.DataFrame, y_test: pd.DataFrame, data: pd.DataFrame, parameters: Dict[str, Any], best_cols):
    mlflow.end_run()
    mlflow.set_tag("mlflow.runName", "prediction decision tree")
    mlflow.autolog()

    if parameters["with_feature_selection"] == False:
        X_test_temp = X_test.copy()
        data_predict = data.drop(columns=parameters["target_column"], axis=1)
    else:
        X_test_temp = X_test[best_cols].copy()
        data_predict = data[best_cols].copy()

    # Calculate model predictions on the test set
    preds = model.predict(X_test_temp)
    pred_labels = np.rint(preds)
    accuracy = sklearn.metrics.f1_score(y_test, pred_labels)

    log = logging.getLogger(__name__)
    log.info("Model f1-score on test set: %0.2f%%", accuracy * 100)

        # Make predictions on the data
    preds = model.predict(data_predict)
    data_predict["prediction"] = preds

    return data_predict



######### Random Forest ###########
def model_predict_rf(model, X_test: pd.DataFrame, y_test: pd.DataFrame, data: pd.DataFrame, parameters: Dict[str, Any], best_cols):
    mlflow.end_run()
    mlflow.set_tag("mlflow.runName", "prediction random forest")
    mlflow.autolog()

    if parameters["with_feature_selection"] == False:
        X_test_temp = X_test.copy()
        data_predict = data.drop(columns=parameters["target_column"], axis=1)
    else:
        X_test_temp = X_test[best_cols].copy()
        data_predict = data[best_cols].copy()

     # Calculate model predictions on the test set
    preds = model.predict(X_test_temp)
    pred_labels = np.rint(preds)
    accuracy = sklearn.metrics.f1_score(y_test, pred_labels)

    log = logging.getLogger(__name__)
    log.info("Model f1-score on test set: %0.2f%%", accuracy * 100)

    mlflow.log_metric("f1_score_rf", accuracy)

    # Make predictions on the data
    preds = model.predict(data_predict)
    data_predict["prediction"] = preds

    return data_predict

