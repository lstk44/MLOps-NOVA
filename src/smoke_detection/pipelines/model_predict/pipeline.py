
"""
This is a boilerplate pipeline
generated using Kedro 0.18.8
"""

from kedro.pipeline import Pipeline, node, pipeline

#from .nodes import  model_predict
from smoke_detection.pipelines.model_predict.nodes import model_predict_rf , model_predict_dt

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            # node(
            #     func=clean_data,
            #     inputs=["dataset2", "parameters"],
            #     outputs=["cleaned_dataset2","raw_describe_dataset2","cleaned_describe_dataset2"],
            #     name="clean_new_data",
            # ),

            # node(
            #     func= feature_engineer,
            #     inputs="cleaned_dataset2",
            #     outputs= "engineered_dataset2",
            #     name="engineering_new_data",
            # ),

            node(
                func= model_predict_dt,
                inputs=["test_model_dt","X_test_data","y_test_data","engineered_dataset2","parameters","best_columns_dt"],
                outputs= "dataset2_prediction_dt",
                name="dataset2_prediction_dt",
            ),

            node(
                func= model_predict_rf,
                inputs=["test_model_rf","X_test_data","y_test_data","engineered_dataset2","parameters","best_columns_rf"],
                outputs= "dataset2_prediction_rf",
                name="dataset2_prediction_rf",
            )
        ]
    )
