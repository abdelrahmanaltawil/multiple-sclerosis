"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.18.3
"""

# env imports
from kedro.pipeline import Pipeline, node, pipeline

# local imports
from multiple_sclerosis.pipelines.data_processing.nodes import extract_data, clean_data, split_data


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=extract_data,
                inputs=["2min_walking_test_paper_data", "params:data"],
                outputs="typed_data",
                name="data_extraction",
            ),
            node(
                func=clean_data,
                inputs="typed_data",
                outputs="cleaned_data",
                name="data_cleaning",
            ),
            node(
                func=split_data,
                inputs=["cleaned_data", "params:data"],
                outputs=["X_train", "y_train", "X_test", "y_test"],
                name="data_splitting",
            ),
        ]
    )