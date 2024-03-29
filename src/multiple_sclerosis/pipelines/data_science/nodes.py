"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.18.3
"""
# env imports
import numpy as np
import pandas as pd
import tensorflow as tf
import plotly.express as px
import sklearn.preprocessing as skl_preprocessing

# local imports
from multiple_sclerosis.pipelines.data_science.helpers.optimizers import get_optimizer
from multiple_sclerosis.pipelines.data_science.helpers.metrics import get_metric
from multiple_sclerosis.pipelines.data_science.helpers.losses import get_loss


def build(neural_network: dict, X_train: pd.DataFrame) -> tf.keras.Model:
    '''
    Build the frame of the neural network. Here the following 
    hyperparameter's of the network are used
    * network depth and spread
    * activation function
    * optimizer
    * LR? @NOTE
    * loss measure
    * quality metrics

    Arguments
    ---------
    * `neural_network`: Parameters define the neural network, found in parameters/data_science.yml.

    Returns
    --------
    `model` : pre-training neural network model
    '''

    num_features = X_train.shape[1]

    model = tf.keras.Sequential(name="DFFNN")

    # neural network layers
    input_layer = tf.keras.layers.InputLayer(input_shape=[num_features])
    hidden_layers = [
        tf.keras.layers.Dense(
            units= neural_network["spread"]*num_features, 
            activation= neural_network["activation"], 
            use_bias = False
            ) 
        for _ in range(neural_network["depth"])
        ]
    output_layer = tf.keras.layers.Dense(units=1, use_bias=False)

    # assembly
    layers = [
        input_layer,
        *hidden_layers,
        output_layer
        ]

    # integration to model
    for layer in layers:
        model.add(layer)

    model.compile( 
        loss= get_loss(neural_network["quality"]["loss"]),
        optimizer= get_optimizer(neural_network["optimizer"]["name"], learning_rate= neural_network["optimizer"]["LR"]),
        metrics= [get_metric(metric) for metric in neural_network["quality"]["metrics"]]
        )
    
    return model


def train_model(model: tf.keras.Model, X_train: pd.DataFrame, y_train: pd.Series, neural_network: dict, normalize_input: bool, callbacks: list = []) -> tuple:
    '''
    Train neural network using the provided samples `training_data` to train the network
    the samples are normalized before feeding to network.
                `training` => normalization => fitting 
    
    Argument
    ----------
    * `model` : neural network model to train
    * `training_data`: Training samples the will tune model weights
    * `parameters`: Parameters defined in parameters.yml.

    Returns
    --------
    `trained_model` : trained neural network model
    '''

    # normalization and scale extraction
    scaler = 0
    if normalize_input:
        X_scaler = skl_preprocessing.MinMaxScaler()
        X_train = X_scaler.fit_transform(X_train)

        y_scaler = skl_preprocessing.MinMaxScaler()
        y_scaler.inverse_transform
        y_train = y_scaler.fit_transform(y_train[:, np.newaxis]).flatten()

        scaler = (X_scaler, y_scaler)

    # model training
    metrics_tracing = model.fit(
                X_train, 
                y_train,
                epochs= neural_network["optimizer"]["epoch"],
                verbose=False,
                callbacks= callbacks
                )

    return model, scaler, pd.DataFrame(metrics_tracing.history)


def test_model(model: tf.keras.Model, X_test: pd.DataFrame, y_test: pd.Series, normalize_input: bool, scaler: object) -> dict:
    '''
    Take testing data to see the model performance and tace the model quality measures
    
    Argument
    ----------
    testing     :   ndarray
                    Testing samples to evaluate the model performance
    '''

    # normalization
    if normalize_input:
        X_scaler, y_scaler = scaler
        X_test = X_scaler.transform(X_test)

    y_predicted = model.predict(X_test).flatten()

    # de-normalization
    if normalize_input:
        y_predicted = y_scaler.inverse_transform(y_predicted[:, np.newaxis]).flatten()
    

    # evaluation
    mae = tf.keras.metrics.mean_absolute_error(
        y_true= y_test,
        y_pred= y_predicted
        )
    rmse = np.sqrt(tf.keras.metrics.mean_squared_error(
        y_true= y_test,
        y_pred= y_predicted
        ))
    
    return {"mae": {"value" : mae, "step": 1}, "rmse": {"value" : rmse, "step": 1}}   


def performance_report(metrics: dict) -> dict:
    '''
    Placeholder
    '''

    print("Performance report: ")
    print("  Mean Absolute Error (mae):", str(float(metrics["model.mae"]["value"])))
    print("  Mean Squared Error (mse):", str(metrics["model.rmse"]["value"]))

    
def performance_visualization(history: pd.DataFrame) -> tuple:
    '''
    Placeholder
    '''

    # plot history
    loss_plot = px.line(history, y='loss')
    
    return loss_plot