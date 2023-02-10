"""Project pipelines."""
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# env imports
from typing import Dict
from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline

# local imports
import multiple_sclerosis.pipelines.hpo.pipeline as hpo
import multiple_sclerosis.pipelines.data_science.pipeline as data_science
import multiple_sclerosis.pipelines.data_processing.pipeline as data_processing


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """

    return {
        "__default__": data_processing.create_pipeline() + data_science.create_pipeline(),
        "data_processing": data_processing.create_pipeline(),
        "hpo": data_processing.create_pipeline() + hpo.create_pipeline()
    }
