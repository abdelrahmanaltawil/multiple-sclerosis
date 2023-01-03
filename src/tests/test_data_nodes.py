# env imports
import yaml
import pytest
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris


# local imports
# from multiple_sclerosis.nodes.data_nodes import *


def test_model_data() -> None:
    '''
    Placeholder
    '''
    raw_data = load_iris()

    # test for empty features or labels, should give all features


    # test for ranges and single features
    features = [
        ["A", "C"],
        ["N"]
    ]

    # assert extract_data(raw_data, parameters=) == 10


def test_split() -> None:
    '''
    Placeholder
    '''

    with pytest.raises(ValueError):
        # write some code that would causes value error
        # let say if training fraction is 0%
        pass



if __name__ == "__main__":


    with open('/parameters.yaml') as open_file:
        
        parameters = yaml.load_all(open_file, Loader=yaml.FullLoader)

        for param in parameters:
            
            for key, value in param.items():
                print(key, "->", value)

    iris = load_iris()
    raw_data = pd.DataFrame(
                    data= np.c_[iris['data'], iris['target']],
                    columns= iris['feature_names'] + ['target']
                    )

    print( raw_data)
