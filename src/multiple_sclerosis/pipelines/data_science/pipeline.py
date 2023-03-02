"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.18.3
"""

# env imports
from kedro.pipeline import Pipeline, node, pipeline

# local imports
from multiple_sclerosis.pipelines.data_science.nodes import build, train_model, test_model, performance_report, performance_visualization


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=build,
                inputs=["params:neural_network", "X_train"],
                outputs="untrained_model",
                name="model_construction",
            ),
            node(
                func=train_model,
                inputs=["untrained_model", "X_train", "y_train", "params:neural_network", "params:data.normalize_input"],
                outputs=["trained_model", "scaler", "training_metrics_history"],
                name="model_training",
            ),
            node(
                func=test_model,
                inputs=["trained_model", "X_test", "y_test", "params:data.normalize_input", "scaler"],
                outputs="performance_metrics",
                name="model_testing",
            ),
            node(
                func=performance_report,
                inputs="performance_metrics",
                outputs= None,
                name="performance_report",
            ),
            node(
                func=performance_visualization,
                inputs="training_metrics_history",
                outputs="loss_history_plot",
            )
        ]
    )