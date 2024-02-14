"""
This is a boilerplate pipeline
generated using Kedro 0.18.8
"""
import logging
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import great_expectations as ge

# Performing data validation using Great Expectations library and logs a message if the data passes the unit tests.
def unit_test(
    data: pd.DataFrame,
): 
    # Convert the pandas dataframe to a great_expectations dataframe
    pd_df_ge = ge.from_pandas(data)

    # Assert column count
    assert pd_df_ge.expect_table_column_count_to_equal(16).success == True
    
    # Assert data types
    assert pd_df_ge.expect_column_values_to_be_of_type("UTC", "int64").success == True
    assert pd_df_ge.expect_column_values_to_be_of_type("Temperature[C]", "float").success == True
    assert pd_df_ge.expect_column_values_to_be_of_type("Humidity[%]", "float").success == True
    assert pd_df_ge.expect_column_values_to_be_of_type("TVOC[ppb]", "int64").success == True
    assert pd_df_ge.expect_column_values_to_be_of_type("eCO2[ppm]", "int64").success == True
    assert pd_df_ge.expect_column_values_to_be_of_type("Raw H2", "int64").success == True
    assert pd_df_ge.expect_column_values_to_be_of_type("Raw Ethanol", "int64").success == True
    assert pd_df_ge.expect_column_values_to_be_of_type("Pressure[hPa]", "float").success == True
    assert pd_df_ge.expect_column_values_to_be_of_type("PM1.0", "float").success == True
    assert pd_df_ge.expect_column_values_to_be_of_type("PM2.5", "float").success == True
    assert pd_df_ge.expect_column_values_to_be_of_type("NC0.5", "float").success == True
    assert pd_df_ge.expect_column_values_to_be_of_type("NC1.0", "float").success == True
    assert pd_df_ge.expect_column_values_to_be_of_type("NC2.5", "float").success == True
    assert pd_df_ge.expect_column_values_to_be_of_type("CNT", "int64").success == True
    assert pd_df_ge.expect_column_values_to_be_of_type("Fire Alarm", "int64").success == True
    
    # Assert column uniqueness to detect duplicates
    assert pd_df_ge.expect_column_values_to_be_unique("UTC").success == True

    # Assert no missing values
    for column in data.columns:
        assert pd_df_ge.expect_column_values_to_not_be_null(column).success == True
    
    # Assert categorical target variable
    assert pd_df_ge.expect_column_values_to_be_in_set("Fire Alarm", [0, 1]).success == True

    # Assert value ranges to detect outliers
    assert pd_df_ge.expect_column_stdev_to_be_between("Temperature[C]", min_value=0, max_value=20).success == True
    assert pd_df_ge.expect_column_stdev_to_be_between("Humidity[%]", min_value=0, max_value=15).success == True
    assert pd_df_ge.expect_column_values_to_be_between("TVOC[ppb]", min_value=0, max_value=None).success == True
    assert pd_df_ge.expect_column_values_to_be_between("eCO2[ppm]", min_value=0, max_value=None).success == True
    assert pd_df_ge.expect_column_mean_to_be_between("Pressure[hPa]", min_value=930, max_value=940)
    assert pd_df_ge.expect_column_values_to_be_between("PM1.0", min_value=0, max_value=None).success == True
    assert pd_df_ge.expect_column_values_to_be_between("PM2.5", min_value=0, max_value=None).success == True
    assert pd_df_ge.expect_column_values_to_be_between("NC0.5", min_value=0, max_value=None).success == True
    assert pd_df_ge.expect_column_values_to_be_between("NC1.0", min_value=0, max_value=None).success == True
    assert pd_df_ge.expect_column_values_to_be_between("NC2.5", min_value=0, max_value=None).success == True
    
    log = logging.getLogger(__name__)
    log.info("Data passed on the unit data tests")

    return 0