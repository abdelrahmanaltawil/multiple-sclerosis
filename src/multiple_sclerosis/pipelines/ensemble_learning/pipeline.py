"""
This is a boilerplate pipeline 'ensemble_learning'
generated using Kedro 0.18.4
"""

# env imports 
from kedro.pipeline import Pipeline, node, pipeline

# local imports
from multiple_sclerosis.pipelines.ensemble_learning.nodes import ensemble_learning, performance_report


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
            node(
                func=ensemble_learning,
                inputs=["X_train", "y_train", "X_test", "y_test", "params:neural_network", "params:data.normalize_input", "params:ensemble_learning"],
                outputs=["ensemble_model", "performance_metrics"],
                name="ensemble_learning",
            ),
            node(
                func=performance_report,
                inputs=["ensemble_model", "performance_metrics"],
                outputs="ensemble_model_report",
                name="visualize",
            )
    ])
