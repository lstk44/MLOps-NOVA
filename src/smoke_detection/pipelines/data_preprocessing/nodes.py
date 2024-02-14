"""
This is a boilerplate pipeline
generated using Kedro 0.18.8
"""

import logging
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import mlflow

def clean_data(data: pd.DataFrame, parameters: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict, Dict]:
    mlflow.end_run()
    """
    This node cleans the data by flooring/capping outliers and imputing missing values
    Input: Raw data
    Output: Cleaned data, dictionary of original summary statistics, dictionary of new, cleaned, summary statistics
    """
    mlflow.set_tag("mlflow.runName", "data cleaning")
    #mlflow.set_tag("mlflow.runName", parameters["run_names"]["run_name_preprocessing"])

    df_transformed = data.copy()

    # drop columns
    df_transformed = df_transformed.drop(columns=['Unnamed: 0', 'UTC', 'CNT'])

    # Get descriptive statistics (pre cleaning)
    describe_to_dict = df_transformed.describe().to_dict()
    
    for col in df_transformed.select_dtypes(include=np.number).columns:
        q1_train, q3_train = np.nanquantile(df_transformed[col], 0.25), np.nanquantile(df_transformed[col], 0.75)
        iqr_train = q3_train - q1_train
        lower_bound_train, upper_bound_train = q1_train-(1.5*iqr_train), q3_train+(1.5*iqr_train)
        df_transformed[col] = [lower_bound_train if i < lower_bound_train else upper_bound_train if i > upper_bound_train else i for i in df_transformed[col]]
        
    
    # Impute missing values with column mean
    df_transformed = df_transformed.fillna(df_transformed.mean())

    # Get new descriptive statistics (post cleaning)
    describe_to_dict_verified = df_transformed.describe().to_dict()

    return df_transformed, describe_to_dict, describe_to_dict_verified 


def feature_engineer(data: pd.DataFrame) -> pd.DataFrame:
    """
    This node creates new features from existing features
    Input: Cleaned data
    Output: Data with new features
    """
    mlflow.end_run()
    mlflow.set_tag("mlflow.runName", "feature engineering")

    df_transformed = data.copy()

    # Add TVOC/eCO2 ratio
    df_transformed['TVOC_eCO2_ratio'] = df_transformed['TVOC[ppb]'] / df_transformed['eCO2[ppm]']

    # One-hot-encode percentiles of some features
    columns = ['Temperature[C]', 'Humidity[%]', 'TVOC[ppb]', 'eCO2[ppm]']
    num_new_columns = 0  # Initialize counter for new columns

    for col in columns:
        percentiles = [0.25, 0.50, 0.75]
        for percentile in percentiles:
            column_name = f"{col}_{int(percentile * 100)}th_percentile"
            percentile_value = df_transformed[col].quantile(percentile)
            df_transformed[column_name] = (df_transformed[col] <= percentile_value).astype(int)
            num_new_columns += 1  # Increment the counter for each new column created

    # Log the number of new columns created
    mlflow.log_metric("num_new_columns", num_new_columns)

    # Log the names of the newly created features
    new_feature_names = df_transformed.columns.tolist()
    mlflow.set_tag("new_feature_names", new_feature_names)

    log = logging.getLogger(__name__)
    log.info(f"The final dataframe has {len(df_transformed.columns)} columns.")

    return df_transformed
