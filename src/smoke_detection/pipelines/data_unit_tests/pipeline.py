"""
This is a boilerplate pipeline
generated using Kedro 0.18.8
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import  unit_test


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=unit_test,
                inputs="dataset1",
                outputs=None,
                name="unit_data_test",
            ),
            node(
                func=unit_test,
                inputs="dataset2",
                outputs=None,
                name="unit_data_test_2",
            ),
        ]
    )


