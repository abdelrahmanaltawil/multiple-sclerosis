# env imports
import tensorflow as tf

def get_loss(loss: str) -> list:
    '''
    Placeholder
    '''

    tensorflow_losses = {
        "mape"  : tf.keras.losses.MeanAbsolutePercentageError(name= "mape"),
        "mae"   : tf.keras.losses.MeanAbsoluteError(name="mae"),
        "mse"   : tf.keras.losses.MeanSquaredError(name= "mse")
    }
    
    if loss not in tensorflow_losses.keys(): raise NotImplemented("loss not implemented")

    return tensorflow_losses[loss]  