# env imports
import tensorflow as tf

def get_metric(metric: str) -> list:
    '''
    Placeholder
    '''

    tensorflow_metrics = {
        "rmse"  : tf.keras.metrics.RootMeanSquaredError(name= "rmse"),
        "mae"   : tf.keras.metrics.MeanAbsoluteError(name= "mae"),
        "mse"   : tf.keras.metrics.MeanSquaredError(name= "mse")
    }

    if metric not in tensorflow_metrics.keys(): raise NotImplemented("metric not implemented")

    return tensorflow_metrics[metric]