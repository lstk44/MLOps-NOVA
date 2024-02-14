import logging
from typing import Any, Dict, Tuple
import mlflow
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE


############ Decision Tree ############
def feature_selection_dt(data: pd.DataFrame, X_train: pd.DataFrame, y_train: pd.DataFrame, parameters: Dict[str, Any]):
    mlflow.end_run()
    """
    This node selects the best features for the model
    Input: Engineered data, X_train, y_train, parameters
    Output: pickle file with the best features
    """
    mlflow.set_tag("mlflow.runName", "feature selection decision tree")
    mlflow.log_param("model", "DecisionTreeClassifier")
    mlflow.log_param("feature_selection", parameters["feature_selection"])

    if parameters["feature_selection"] == "rfe":
        model = DecisionTreeClassifier(max_depth=parameters["dt"]["max_depth"], min_samples_split=parameters["dt"]["min_samples_split"])
        model.fit(X_train, y_train.iloc[:, 0].values)
        rfe = RFE(model)
        rfe = rfe.fit(X_train, y_train.iloc[:, 0].values)
        f = rfe.get_support(1)
        X_cols = X_train.columns[f].tolist()

        mlflow.log_param("best_columns_dt", X_cols)

    log = logging.getLogger(__name__)
    log.info(f"Number of best columns for DecisionTreeClassifier: {len(X_cols)}")

    mlflow.log_metric("num_best_columns_dt", len(X_cols))

    return X_cols

############ Random Forest ############
def feature_selection_rf(data: pd.DataFrame, X_train: pd.DataFrame, y_train: pd.DataFrame, parameters: Dict[str, Any]):
    mlflow.end_run()
    """
    This node selects the best features for the model
    Input: Engineered data, X_train, y_train, parameters
    Output: pickle file with the best features
    """
    mlflow.set_tag("mlflow.runName", "feature selection random forest")
    mlflow.log_param("model2", "RandomForestClassifier")
    mlflow.log_param("feature_selection", parameters["feature_selection"])

    if parameters["feature_selection"] == "rfe":
        model = RandomForestClassifier(
            n_estimators=parameters["rf"]["n_estimators"],
            max_depth=parameters["rf"]["max_depth"],
            max_features=parameters["rf"]["max_features"]
        )
        model.fit(X_train, y_train.iloc[:, 0].values)
        rfe = RFE(model)
        rfe = rfe.fit(X_train, y_train.iloc[:, 0].values)
        f = rfe.get_support(1)
        X_cols = X_train.columns[f].tolist()

        mlflow.log_param("best_columns_rf", X_cols)

    log = logging.getLogger(__name__)
    log.info(f"Number of best columns for RandomForestClassifier: {len(X_cols)}")

    mlflow.log_metric("num_best_columns_rf", len(X_cols))

    return X_cols