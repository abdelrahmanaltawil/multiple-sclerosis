"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.18.3
"""

# env imports
from kedro.pipeline import Pipeline, node, pipeline

# local imports
from multiple_sclerosis.nodes.data_nodes import extract_data, clean, split


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            # data nodes
            node(
                func=extract_data,
                inputs=["2min_walking_test_data", "parameters"],
                outputs="modeling_data",
                name="data_extraction",
            ),
            node(
                func=clean,
                inputs=["modeling_data", "parameters"],
                outputs="cleaned_data",
                name="data_cleaning",
            ),
            node(
                func=split,
                inputs=["cleaned_data", "parameters"],
                outputs=["X_train", "X_test", "y_train", "y_test"],
                name="data_splitting",
            )
        ]
    )