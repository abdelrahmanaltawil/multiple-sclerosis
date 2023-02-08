# env imports
import tensorflow as tf

def get_metric(metric: str) -> list:
    '''
    Placeholder
    '''

    tensorflow_metrics = {
    "mae"   : tf.keras.metrics.MeanAbsoluteError(name= "mae"),
    "mse"   : tf.keras.metrics.MeanSquaredError(name= "mse"),
    "rmse"  : tf.keras.metrics.RootMeanSquaredError(name= "rmse")
    }

    if metric not in tensorflow_metrics.keys(): raise NotImplemented("metric not implemented")

    return tensorflow_metrics[metric]