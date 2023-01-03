"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.18.3
"""

# env imports
import pandas as pd

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


def clean_data(data: pd.DataFrame, parameters: dict) -> pd.DataFrame:
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