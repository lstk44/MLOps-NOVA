#"""Project pipelines."""
#from typing import Dict

#from kedro.framework.project import find_pipelines
#from kedro.pipeline import Pipeline


#def register_pipelines() -> Dict[str, Pipeline]:
#    """Register the project's pipelines.

#    Returns:
#        A mapping from pipeline names to ``Pipeline`` objects.
#    """
#    pipelines = find_pipelines()
#    pipelines["__default__"] = sum(pipelines.values())
#    return pipelines

import kedro
import mlflow

"""Project pipelines."""
from typing import Dict
from kedro.pipeline import Pipeline, pipeline


from smoke_detection.pipelines import (
    data_preprocessing as preprocessing,
    data_split as split_data,
    model_train as train,
    feature_selection as best_features,
    model_predict as predict,
    data_drift as drift_test,
    data_unit_tests as unit_test

)

def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.
    """
    preprocessing_stage = preprocessing.create_pipeline()
    split_data_stage = split_data.create_pipeline()
    train_stage = train.create_pipeline()
    feature_selection_stage = best_features.create_pipeline()
    predict_stage = predict.create_pipeline()
    drift_test_stage = drift_test.create_pipeline()
    unit_test_stage = unit_test.create_pipeline()

    return {
        "preprocessing": preprocessing_stage,
        "split_data": split_data_stage,
        "train": train_stage,
        "feature_selection": feature_selection_stage,
        "predict": predict_stage,
        "drift_test" : drift_test_stage, 
        "unit_test": unit_test_stage,
        # Full Pipeline
        "__default__": unit_test_stage + preprocessing_stage + split_data_stage + feature_selection_stage + train_stage + drift_test_stage + predict_stage
        }

#if __name__ == "__main__":
    mlflow.autolog()  # Enable automatic logging with MLflow

    pipelines = register_pipelines()
    kedro.runner.SequentialRunner().run(pipelines)