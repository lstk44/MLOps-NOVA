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
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
import sklearn
import mlflow
import nannyml as nml

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

def data_drift(reference_data: pd.DataFrame, analysis_data: pd.DataFrame):

    # define column names
    column_names = reference_data.columns.tolist()

    ###################### Univariate Drift #############################
    # generate a report for some numeric features using KS test and evidely ai
    data_drift_report = Report(metrics=[
        DataDriftPreset(cat_stattest='ks', stattest_threshold=0.05)])

    data_drift_report.run(current_data=analysis_data[column_names],
                          reference_data=reference_data[column_names],
                          column_mapping=None)

    # save report
    data_drift_report.save_html("data/08_reporting/univariate_data_drift_report.html")

    ###################### MultiVariate Drift #############################
    # Let's initialize the object that will perform the Univariate Drift calculations
    multivariate_calculator = nml.DataReconstructionDriftCalculator(
        column_names= column_names,
        chunk_number = 10,
        imputer_categorical=sklearn.impute.SimpleImputer(strategy='constant', fill_value='missing'),
        imputer_continuous=sklearn.impute.SimpleImputer(strategy='median'))

    multivariate_calculator.fit(reference_data)
    results = multivariate_calculator.calculate(analysis_data)
    results_df = results.to_df()

    # multivariate_drift = results.plot()

    return results_df # , multivariate_drift