"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.18.3
"""

# env imports
import pandas as pd
import sklearn.model_selection as skl_model_selection

# local imports


def extract_data(raw_data: pd.DataFrame, data_params: dict) -> pd.DataFrame:
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
    for feature_a, feature_b in data_params["features_columns"]:
        columns.append(raw_data.loc[:, feature_a : feature_b])

    columns.append(raw_data.loc[:, data_params["target_column"]])

    model_data = pd.concat(columns, axis=1)    

    return model_data


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
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


def split_data(data: pd.DataFrame, data_params: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
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

    target_data = data.pop(data_params["target_column"]).values
    features_data = data.values

    # split data
    X_train, X_test, y_train, y_test = skl_model_selection.train_test_split(
                            features_data, 
                            target_data, 
                            train_size= data_params["train_fraction"],
                            random_state= data_params["random_state"]
                            )


    return X_train, y_train, X_test, y_test