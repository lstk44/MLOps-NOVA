"""
This is a boilerplate pipeline
generated using Kedro 0.18.8
"""

import logging
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import mlflow
from sklearn.model_selection import train_test_split


def split_data(data: pd.DataFrame, parameters: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    mlflow.end_run()
    """
    This node splits the data into training and test sets
    Input: Engineered data, parameters
    Output: X_train, X_test, y_train, y_test
    """
    mlflow.set_tag("mlflow.runName", "splitting data")

    # Log parameters
    mlflow.log_param("target_column", parameters["target_column"])
    mlflow.log_param("test_fraction", parameters["test_fraction"])
    mlflow.log_param("random_state", parameters["random_state"])

    # Check for nulls
    assert [col for col in data.columns if data[col].isnull().any()] == []

    # Split columns and target
    x, y = data.drop(columns=parameters["target_column"], axis=1), data[parameters["target_column"]]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=parameters["test_fraction"], random_state=parameters["random_state"])

    # Log metrics
    mlflow.log_metric("train_set_size", len(X_train))
    mlflow.log_metric("test_set_size", len(X_test))

    return X_train, X_test, y_train, y_test
