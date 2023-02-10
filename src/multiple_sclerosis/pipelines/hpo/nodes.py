"""
This is a boilerplate pipeline 'hpo'
generated using Kedro 0.18.4
"""

# env imports
import optuna
import numpy as np
import pandas as pd
import tensorflow as tf
from optuna.trial import TrialState
from optuna.visualization import plot_contour
from optuna.integration import TFKerasPruningCallback
from optuna.visualization import plot_param_importances
from optuna.visualization import plot_intermediate_values
from optuna.visualization import plot_parallel_coordinate
from optuna.visualization import plot_optimization_history

# local imports
from multiple_sclerosis.pipelines.data_science.nodes import build, train_model, test_model


def hyperparameters_optimization(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series , neural_network: dict, normalize_input: bool, hpo: dict) -> None:
    '''
    Placeholder
    '''

    def objective(trial):
        '''
        Placeholder
        '''
        # clear clutter from previous TensorFlow graphs.
        tf.keras.backend.clear_session()
        # tf.compat.v1.disable_v2_behavior()

        # candidate in from the search space 
        neural_network["spread"] = trial.suggest_int("spread", *hpo["spread_range"])
        neural_network["depth"] = trial.suggest_int("depth", *hpo["depth_range"])
        neural_network["optimizer"]["name"] = trial.suggest_categorical("optimizer", hpo["optimizers"])
        neural_network["optimizer"]["LR"] = trial.suggest_float("LR", *hpo["learning_rate_range"], log=True)
        neural_network["activation"] = trial.suggest_categorical("activation", hpo["activation_functions"])

        with tf.device('/CPU:0'):
   
            model = build(neural_network)
            # trained_model, scalers, history = train_model(model, X_train, y_train, neural_network, normalize_input)
            # metrics = test_model(trained_model, X_test, y_test, normalize_input, scalers)


            callbacks = [
                tf.keras.callbacks.EarlyStopping(monitor="val_rmse", patience=300, mode="min"),
                TFKerasPruningCallback(trial, "val_rmse"),
                ]

            # Train model.
            history = model.fit(
                X_train,
                y_train,
                epochs= neural_network["optimizer"]["epoch"],
                validation_data= (X_test, y_test),
                verbose= False,
                callbacks= callbacks,
                )


        return  history.history["val_rmse"][-1]

    study = optuna.create_study(
        study_name= hpo["study_name"], 
        sampler= optuna.samplers.NSGAIISampler(),
        pruner= optuna.pruners.SuccessiveHalvingPruner(),
        direction= hpo["optimization_direction"]
        )
    
    study.optimize(
        func= objective, 
        n_trials=500
        )

    return study


def study_report(study: optuna.Study) -> dict:
    '''
    Placeholder
    '''

    best_model_hyperparameters = study.best_params

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))    

    return best_model_hyperparameters


def study_visualization(study: optuna.Study) -> None:
    '''
    Placeholder
    '''

    optimization_history_plot = plot_optimization_history(study)
    # plot_intermediate_values(study)
    parallel_coordinate_plot = plot_parallel_coordinate(study)
    # plot_contour(study)
    param_importances_plot = plot_param_importances(study)

    return optimization_history_plot, parallel_coordinate_plot, param_importances_plot