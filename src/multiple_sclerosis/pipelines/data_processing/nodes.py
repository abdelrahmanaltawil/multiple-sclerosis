"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.18.3
"""

# env imports
import pandas as pd
import sklearn.model_selection as skl

# local imports


def extract_data(raw_data: pd.DataFrame, parameters: dict) -> pd.DataFrame:
    '''
    Extract model data that will used for processing of the neural network

    Arguments
    ---------
    * `raw_data`: The original, immutable data.
    * `parameters`: Parameters defined in parameters.yml.
    
    Returns
    --------
    `model_data` : data used for constructing the machine learning model
    '''

    columns=[]
    for feature_a, feature_b in parameters["data"]["features_columns"]:
        columns.append(raw_data.loc[:, feature_a : feature_b])

    columns.append(raw_data.loc[:, parameters["data"]["target_column"]])

    model_data = pd.concat(columns, axis=1)    

    return model_data


def clean(data: pd.DataFrame, parameters: dict) -> pd.DataFrame:
    '''
    This method by no means meant to be comprehensive, however the user have to decide what cleaning schemes to use

    Arguments
    ----------
    * `data`: Data containing features and target.
    * `parameters`: Parameters defined in parameters.yml.
    
    Return
    -------
    `cleaned_data` : data after applying the given cleaning criterions
    ''' 

    # drop samples contains nan values
    cleaned_data = data.dropna()
    
    return cleaned_data


def split(data: pd.DataFrame, parameters: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
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


    y = data.pop(parameters["data"]["target_column"])
    X = data

    # split data
    X_train, X_test, y_train, y_test = skl.train_test_split(
                            X, 
                            y, 
                            train_size=parameters["data"]["train_fraction"],
                            random_state=parameters["data"]["random_state"]
                            )


    return X_train, X_test, y_train, y_test