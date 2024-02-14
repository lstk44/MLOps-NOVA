
"""
This is a boilerplate pipeline
generated using Kedro 0.18.8
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import data_drift
from smoke_detection.pipelines.data_preprocessing.nodes import clean_data, feature_engineer

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
                func= data_drift,
                inputs=["engineered_dataset2","engineered_dataset1",],
                outputs= "multivariate_drift_results",
                name="drift_analysis",
            ),
        ]
    )