"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.18.3
"""
# env imports
import numpy as np
import pandas as pd
import tensorflow as tf
import plotly.express as px
from tabulate import tabulate
from sklearn.metrics import r2_score
import sklearn.preprocessing as skl_preprocessing

# local imports
from multiple_sclerosis.pipelines.data_science.helpers.optimizers import get_optimizer
from multiple_sclerosis.pipelines.data_science.helpers.metrics import get_metric
from multiple_sclerosis.pipelines.data_science.helpers.losses import get_loss


def build(neural_network: dict, features_count: int = 92) -> tf.keras.Model:
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

    model = tf.keras.Sequential()

    model.add(
        tf.keras.layers.Dense(
            neural_network["spread"]*features_count, 
            activation= neural_network["activation"], 
            use_bias = False,
            # input_shape=[features_count]
            input_shape=(features_count,)
            )
        )

    for i in range(1, neural_network["depth"]):
        model.add(
            tf.keras.layers.Dense(
                    neural_network["spread"]*features_count, 
                    use_bias= False,
                    activation= neural_network["activation"]
                    )
                )

    model.add(tf.keras.layers.Dense(1))


    model.compile( 
        loss= get_loss(neural_network["quality"]["loss"]),
        optimizer= get_optimizer(neural_network["optimizer"]["name"], learning_rate= neural_network["optimizer"]["LR"]),
        metrics= [get_metric(metric) for metric in neural_network["quality"]["metrics"]]
        )
    
    print(model.summary())

    return model


def train_model(model: tf.keras.Model, X_train: pd.DataFrame, y_train: pd.Series, neural_network: dict, normalize_input: bool) -> tuple:
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
    if normalize_input:
        X_scaler = skl_preprocessing.MinMaxScaler()
        X_train = X_scaler.fit_transform(X_train)

    # model training
    metrics_tracing = model.fit(
                X_train, 
                y_train,
                epochs= neural_network["optimizer"]["epoch"],
                verbose=False
                )

    return model, X_scaler, pd.DataFrame(metrics_tracing.history)


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
        X_test = scaler.transform(X_test)

    y_predicted = model.predict(X_test).flatten()

    # evaluation
    mae = tf.keras.metrics.mean_absolute_error(
        y_true= y_test,
        y_pred= y_predicted
        )
    rmse = np.sqrt(tf.keras.metrics.mean_squared_error(
        y_true= y_test,
        y_pred= y_predicted
        ))
    # r_square = r2_score(
    #     y_true= y_test,
    #     y_pred= y_predicted
    # )
    r_square = 0
    
    return {"mae": mae, "rmse": rmse, "r_square": r_square}   


def performance_report(metrics: dict) -> dict:
    '''
    Placeholder
    '''

    data = [
        ["Mean Absolute Error (mae)", str(float(metrics["mae"]))],
        ["Mean Squared Error (mse)", str(metrics["rmse"])],
        ["R^2 (r_square)", str(metrics["r_square"])]
        ]
    
    print("\nPERFORMANCE REPORT")
    print(tabulate(data , headers=["Metric", "Value"], tablefmt='rst'),'\n')


def performance_visualization(history: pd.DataFrame) -> tuple:
    '''
    Placeholder
    '''

    # plot history
    metrics_plot = px.line(history, y=['loss', 'rmse', 'mae'])
    
    return metrics_plot


# def create_confusion_matrix(y_test, y_predicted):
#     '''
#     Placeholder
#     '''

#     data = {"y_Actual": y_test, "y_Predicted": y_predicted}
#     df = pd.DataFrame(data, columns=["y_Actual", "y_Predicted"])
#     confusion_matrix = pd.crosstab(
#         df["y_Actual"], df["y_Predicted"], rownames=["Actual"], colnames=["Predicted"]
#     )
#     plt = sn.heatmap(confusion_matrix, annot=True)

#     mlflow.log_figure(plt, "figure")
    
#     return plt

