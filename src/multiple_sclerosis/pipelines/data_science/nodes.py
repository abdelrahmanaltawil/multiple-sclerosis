"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.18.3
"""

# env imports
import pandas as pd
import tensorflow as tf
import sklearn.model_selection as skl

# local imports


def build(parameters: dict) -> tf.keras.Model:
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
    * `parameters`: Parameters defined in parameters.yml.
    
    Returns
    --------
    `model` : pre-training neural network model
    '''

    model = tf.keras.Sequential()

    model.add(
        tf.keras.layers.Dense(
            parameters["neural_network"]["spread"]*90, 
            activation=parameters["neural_network"]["activation"], 
            use_bias = False,
            input_shape=[90]
            )
        )

    for i in range(1, parameters["neural_network"]["depth"]):
        model.add(
            tf.keras.layers.Dense(
                    parameters["neural_network"]["spread"]*90, 
                    use_bias=False,
                    activation=parameters["neural_network"]["activation"]
                    )
                )

    model.add(tf.keras.layers.Dense(1))

    model.compile( 
        loss=parameters["neural_network"]["quality"]["loss"],
        optimizer=parameters["neural_network"]["optimizer"]["name"],
        metrics=parameters["neural_network"]["quality"]["metrics"]
        )
    
    return model


def split_data(data: pd.DataFrame, parameters: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    '''
    Splits the data to training and test portions based on the parameters provided in 
    `multiple-sclerosis/conf/base/parameters`
    
    Arguments
    ---------
    * `data`: Data containing features and target.
    * `parameters`: Parameters defined in parameters.yml.
    
    Returns
    --------
    `X_train`, `X_test`, `y_train`, `y_test`: training and testing data
    '''

    target_data =  data.pop(parameters["data"]["target_column"])
    features_data = data

    # split data
    X_train, X_test, y_train, y_test = skl.train_test_split(
                            features_data, 
                            target_data, 
                            train_size=parameters["data"]["train_fraction"],
                            random_state=parameters["data"]["random_state"]
                            )


    return X_train, X_test, y_train, y_test


def train_model(model: tf.keras.Model, X_train: pd.DataFrame, y_train: pd.DataFrame, parameters: dict) -> tf.keras.Model:
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

    # data_min, data_max = training_data.min(), training_data.max()

    normalize = lambda data : (data - data.min()) / (data.max() - data.min())

    X_train = normalize(X_train)
    y_train = normalize(y_train)


    history = model.fit(
                X_train, 
                y_train,
                epochs = parameters["neural_network"]["optimizer"]["epoch"],
                verbose=0
                )

    return model


def test(paramters) -> None:
    '''
    placeholder
    '''

    print("hi")