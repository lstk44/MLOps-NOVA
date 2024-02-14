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



############### Decision Tree #################
def model_train_dt(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.DataFrame, y_test: pd.DataFrame, parameters: Dict[str, Any], best_cols):
    mlflow.end_run()
    
    if parameters["with_feature_selection"] == False:
        X_train_temp = X_train.copy()
        X_test_temp = X_test.copy()
        best_cols = list(X_train.columns)
    else:
        X_train_temp = X_train[best_cols].copy()
        X_test_temp = X_test[best_cols].copy()
    
    mlflow.set_tag("mlflow.runName", "training decision tree")
    mlflow.autolog()

    model = DecisionTreeClassifier(max_depth=parameters["dt"]["max_depth"], min_samples_split=parameters["dt"]["min_samples_split"])
    model.fit(X_train_temp, y_train.iloc[:, 0].values)

        # Create object that can calculate shap values
    explainer = shap.TreeExplainer(model)

        # calculate shap values. This is what we will plot.
    shap_values = explainer.shap_values(X_test)

    shap.summary_plot(shap_values[1], X_test, show=False)

    preds = model.predict(X_test_temp)
    pred_labels = np.rint(preds)
    accuracy = sklearn.metrics.f1_score(y_test.iloc[:, 0].values, pred_labels)

    log = logging.getLogger(__name__)
    log.info(f"Number of best columns: {len(best_cols)}")
    log.info("Decision Tree Model f1-score on test set: %0.2f%%", accuracy * 100)

    return model, plt


############### Random Forest #################
def model_train_rf(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.DataFrame, y_test: pd.DataFrame, parameters: Dict[str, Any], best_cols):
    mlflow.end_run()

    if parameters["with_feature_selection"] == False:
        X_train_temp = X_train.copy()
        X_test_temp = X_test.copy()
        best_cols = list(X_train.columns)
    else:
        X_train_temp = X_train[best_cols].copy()
        X_test_temp = X_test[best_cols].copy()

    mlflow.set_tag("mlflow.runName", "training random forest")
    mlflow.autolog()

    model = RandomForestClassifier(
        n_estimators=parameters["rf"]["n_estimators"],
        max_depth=parameters["rf"]["max_depth"],
        max_features=parameters["rf"]["max_features"]
        )
    model.fit(X_train_temp, y_train.iloc[:, 0].values)

        # Create object that can calculate shap values
    explainer = shap.TreeExplainer(model)

        # calculate shap values. This is what we will plot.
    shap_values = explainer.shap_values(X_test)

    shap.summary_plot(shap_values[1], X_test, show=False)

    preds = model.predict(X_test_temp)
    pred_labels = np.rint(preds)
    accuracy = sklearn.metrics.f1_score(y_test.iloc[:, 0].values, pred_labels)

    log = logging.getLogger(__name__)
    log.info(f"Number of best columns: {len(best_cols)}")
    log.info("Random Forest Model f1-score on test set: %0.2f%%", accuracy * 100)

    return model, plt






