"""
This module contains an example test.

Tests should be placed in ``src/tests``, in modules that mirror your
project's structure, and in files named test_*.py. They are simply functions
named ``test_*`` which test a unit of logic.

To run the tests, run ``kedro test`` from the project root directory.
"""

from pathlib import Path
import pickle

import pytest
import pandas as pd
import numpy as np

from kedro.framework.project import settings
from kedro.config import ConfigLoader
from kedro.framework.context import KedroContext
from kedro.framework.hooks import _create_hook_manager
import sys
import os

#sys.path.append("C:/Users/rosan/Desktop/class-bank-example/src/class_bank_example/pipelines/")
from smoke_detection.pipelines.data_preprocessing.nodes import clean_data, feature_engineer
from smoke_detection.pipelines.data_split.nodes import split_data
from smoke_detection.pipelines.feature_selection.nodes import feature_selection_rf, feature_selection_dt
# from smoke_detection.pipelines.data_drift.nodes import data_drift
from smoke_detection.pipelines.model_predict.nodes import model_predict_dt, model_predict_rf
from smoke_detection.pipelines.model_train.nodes import model_train_rf, model_train_dt


# Data Preprocessing
df_raw = pd.read_csv("data/01_raw/smoke_detection_1.csv")
parameters = {
    "test_fraction": 0.3,
    "random_state": 2021,
    "target_column": "Fire Alarm",
    "feature_selection": "rfe",
    "model_name": "smoke_detection",
    "with_feature_selection": True,
    "dt": {
        "max_depth": 6,
        "min_samples_split": 2
    },
    "rf": {
        "n_estimators": 100,
        "max_depth": 6,
        "max_features": 3
    }
}

def test_cleaned_data():
    """
    Test whether the variable describe_to_dict returned from the clean_data() function is an instance of the dict class
    """
    df_transformed, describe_to_dict, describe_to_dict_verified = clean_data(df_raw, parameters)
    isinstance(describe_to_dict, dict)

def test_null_values():
    """
    Test if all empty values were imputed
    """
    df_transformed, describe_to_dict, describe_to_dict_verified = clean_data(df_raw, parameters)
    assert [col for col in df_transformed.columns if df_transformed[col].isnull().any()] == []

# def test_check_duplicates():
#     """
#     Assert that there are no duplicates in the data
#     """
#     df_transformed, describe_to_dict, describe_to_dict_verified = clean_data(df_raw, parameters)
#     duplicate_rows = df_transformed[df_transformed.duplicated()]
#     assert len(duplicate_rows) == 0

def test_drop_column():
    """
    Test if all empty values were imputed
    """
    df_transformed, describe_to_dict, describe_to_dict_verified = clean_data(df_raw, parameters)
    assert 'Unnamed: 0' not in df_transformed.columns
    assert 'UTC' not in df_transformed.columns
    assert 'CNT' not in df_transformed.columns


# Data Engineering
df_cleaned = pd.read_csv("data/02_intermediate/smoke_detection_1_cleaned.csv")

def test_create_column():
    """
    Test if the new variable got successfully created
    """
    df_final = feature_engineer(df_cleaned)
    assert ('TVOC_eCO2_ratio' in df_final.columns)


# Data Split
df_engineered = pd.read_csv("data/04_feature/engineered_dataset1.csv")

def test_split_data():
    """
    Test the split_data function
    """
    X_train, X_test, y_train, y_test = split_data(df_engineered, parameters)

    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(X_test, pd.DataFrame)
    assert isinstance(y_train, pd.Series)
    assert isinstance(y_test, pd.Series)


# Data Drift
# data_reference = pd.read_csv("data/04_feature/engineered_dataset1.csv")
# data_analysis = pd.read_csv("data/04_feature/engineered_dataset2.csv")

# def test_data_drift():
#     """
#     Test the data_drift function
#     """
#     drift_results = data_drift(data_reference, data_analysis)
#     assert isinstance(drift_results, pd.DataFrame)


# Feature Selection
X_train_data = pd.read_csv("data/05_model_input/X_train.csv")
y_train_data = pd.read_csv("data/05_model_input/y_train.csv")

def test_feature_selection_rf():
    """
    Test the feature_selection_rf function
    """
    selected_features = feature_selection_rf(df_engineered, X_train_data, y_train_data, parameters)
    assert isinstance(selected_features, list)


def test_feature_selection_dt():
    """
    Test the feature_selection_dt function
    """
    selected_features = feature_selection_dt(df_engineered, X_train_data, y_train_data, parameters)
    assert isinstance(selected_features, list)

# Model Predictions 
X_test = pd.read_csv("data/05_model_input/X_test.csv")
y_test = pd.read_csv("data/05_model_input/y_test.csv")

def test_model_predict_rf():
    """
    Test the model prediction using a Random Forest Classifier
    """
    with open("data/06_models/trained_model_rf.pkl", "rb") as file:
        model = pickle.load(file)

    with open("data/06_models/best_cols_rf.pkl", "rb") as file:
        best_cols = pickle.load(file)

    result = model_predict_rf(model, X_test, y_test, df_engineered, parameters, best_cols)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == len(df_engineered)
    assert 'prediction' in result.columns

def test_model_predict_dt():
    """
    Test the model prediction using a Decision Tree Classifier
    """
    with open("data/06_models/trained_model_dt.pkl", "rb") as file:
        model = pickle.load(file)

    with open("data/06_models/best_cols_dt.pkl", "rb") as file:
        best_cols = pickle.load(file)

    result = model_predict_rf(model, X_test, y_test, df_engineered, parameters, best_cols)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == len(df_engineered)
    assert 'prediction' in result.columns