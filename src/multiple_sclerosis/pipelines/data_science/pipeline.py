"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.18.3
"""

# env imports
from kedro.pipeline import Pipeline, node, pipeline

# local imports
from multiple_sclerosis.pipelines.data_science.nodes import build, split_data, train_model, test


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=build,
                inputs="parameters",
                outputs="model",
                name="model_construction_node",
            ),
            node(
                func=split_data,
                inputs=["cleaned_data", "parameters"],
                outputs=["X_train", "X_test", "y_train", "y_test"],
                name="data_splitting_node",
            ),
            node(
                func=train_model,
                inputs=["model", "X_train", "y_train", "parameters"],
                outputs="trained_model",
                name="model_training_node",
            )
        ]
    )