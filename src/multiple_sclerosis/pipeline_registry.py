"""Project pipelines."""

# env imports
from typing import Dict
from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline

# local imports
import multiple_sclerosis.pipelines.data_processing.pipeline as dp
import multiple_sclerosis.pipelines.data_science.pipeline as ds


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    # pipelines = find_pipelines()
    # pipelines["__default__"] = sum(pipelines.values())
    # return pipelines

    data_processing_pipeline = dp.create_pipeline()
    data_science_pipeline = ds.create_pipeline()

    return {
        "__default__": data_processing_pipeline + data_science_pipeline,
        "data_processing": data_processing_pipeline,
        "data_science": data_science_pipeline,
    }
