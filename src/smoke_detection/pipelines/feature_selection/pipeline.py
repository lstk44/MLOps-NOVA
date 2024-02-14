
"""
This is a boilerplate pipeline
generated using Kedro 0.18.8
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import feature_selection_dt, feature_selection_rf


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=feature_selection_dt,
                inputs=["engineered_dataset1","X_train_data","y_train_data","parameters"],
                outputs="best_columns_dt",
                name="feature_selection_dt"
            ),
            
            node(
                func=feature_selection_rf,
                inputs=["engineered_dataset1","X_train_data","y_train_data","parameters"],
                outputs="best_columns_rf",
                name="feature_selection_rf"
            )
        ]
    )
