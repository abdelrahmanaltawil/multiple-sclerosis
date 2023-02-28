"""
This is a boilerplate pipeline 'ensemble_learning'
generated using Kedro 0.18.4
"""

# env imports
import functools
import numpy as np
import pandas as pd
import tensorflow as tf
import sklearn.ensemble
from sklearn.svm import SVR
from sklearn.linear_model import Lasso, LinearRegression
from scikeras.wrappers import KerasRegressor
from sklearn.tree import DecisionTreeRegressor
import sklearn.preprocessing as skl_preprocessing
from sklearn.ensemble import BaggingRegressor, AdaBoostRegressor, StackingRegressor

# local imports
from multiple_sclerosis.pipelines.data_science.nodes import build, train_model, test_model


def ensemble_learning(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, 
                      y_test: pd.Series, neural_network: dict, normalize_input: bool, 
                      ensemble_learning: dict) -> None:
    '''
    Placeholder
    '''
    
    create_model = functools.partial(
        build,
        neural_network= neural_network, 
        X_train= X_train
        )

    # casting the model to skl
    model = KerasRegressor(
                    model= create_model,
                    epochs= neural_network["optimizer"]["epoch"],
                    verbose= False
                    )

    # ensemble method
    if ensemble_learning["method"] == "Bagging":
        ensemble = BaggingRegressor(
                        estimator= model, 
                        n_estimators= ensemble_learning["n_estimators"]
                        )

    elif ensemble_learning["method"] == "Boosting":
        ensemble = AdaBoostRegressor(
                        estimator= model, 
                        n_estimators= ensemble_learning["n_estimators"]
                        )

    elif ensemble_learning["method"] == "Stacking":
        ensemble = StackingRegressor(
                        estimators= [
                                ("Neural Network", model),
                                ("Decision Tree Regressor", DecisionTreeRegressor()),
                                ("Lasso", Lasso()),
                                ("Support Vector Regressor", SVR())
                                ],
                        final_estimator= LinearRegression()
                        )


    if normalize_input:
        X_scaler = skl_preprocessing.MinMaxScaler()
        X_train = X_scaler.fit_transform(X_train)
        X_test = X_scaler.transform(X_test)

    # training
    ensemble.fit(X_train, y_train)

    # testing
    y_predicted = ensemble.predict(X_test)

    mae = tf.keras.metrics.mean_absolute_error(
        y_true= y_test,
        y_pred= y_predicted
        )
    rmse = np.sqrt(tf.keras.metrics.mean_squared_error(
        y_true= y_test,
        y_pred= y_predicted
        ))

    return ensemble, {"mae": {"value" : mae, "step": 1}, "rmse": {"value" : rmse, "step": 1}}  


def performance_report(ensemble: sklearn.ensemble.BaseEnsemble, metrics: dict) -> dict:
    '''
    Placeholder
    '''
    
    print("Performance report: ")
    print("  Mean Absolute Error (mae):", str(float(metrics["model.mae"]["value"])))
    print("  Mean Squared Error (mse):", str(metrics["model.rmse"]["value"]))

    return ensemble._repr_html_()
