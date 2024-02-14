
"""
This is a boilerplate pipeline
generated using Kedro 0.18.8
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import  model_train_dt, model_train_rf


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func= model_train_dt,
                inputs=["X_train_data","X_test_data","y_train_data","y_test_data","parameters","best_columns_dt"],
                outputs= ["test_model_dt","output_plot_dt"],
                name="train_dt"
            ),
            
            node(
                func= model_train_rf,
                inputs=["X_train_data","X_test_data","y_train_data","y_test_data","parameters","best_columns_rf"],
                outputs= ["test_model_rf","output_plot_rf"],
                name="train_rf"
            )
        ]
    )
