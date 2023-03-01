"""
This is a boilerplate pipeline 'hpo'
generated using Kedro 0.18.4
"""

# env imports
from kedro.pipeline import Pipeline, node, pipeline

# local imports
from multiple_sclerosis.pipelines.hpo.nodes import hyperparameters_optimization, study_report, study_visualization


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
            node(
                func=hyperparameters_optimization,
                inputs=["X_train", "y_train", "X_test", "y_test", "params:neural_network", "params:data.normalize_input", "params:hpo"],
                outputs="study",
                name="hyperparameters_optimization",
            ),
            node(
                func=study_report,
                inputs="study",
                outputs= ["best_model_parameters", "performance_metrics", "trials_dataframe"],
                name="study_report",
            ),
            node(
                func=study_visualization,
                inputs="study",
                outputs=["optimization_history_plot", "intermediate_values_plot", "parallel_coordinate_plot", "param_importances_plot", "contour_plot"],
                name="study_visualization",
            )
    ])
