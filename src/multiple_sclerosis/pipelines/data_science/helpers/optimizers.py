# env imports
import tensorflow as tf


def get_optimizer(optimizer: str, learning_rate: float) -> tf.keras.optimizers:
    '''
    Placeholder
    '''

    tensorflow_optimizers = {
        "Adadelta"  : tf.keras.optimizers.legacy.Adadelta(learning_rate= learning_rate),
        "RMSprop"   : tf.keras.optimizers.legacy.RMSprop(learning_rate= learning_rate),
        "Adagrad"   : tf.keras.optimizers.legacy.Adagrad(learning_rate= learning_rate),
        "Adamax"    : tf.keras.optimizers.legacy.Adamax(learning_rate= learning_rate),
        "Nadam"     : tf.keras.optimizers.legacy.Nadam(learning_rate= learning_rate),
        "Adam"      : tf.keras.optimizers.legacy.Adam(learning_rate= learning_rate),
        "Ftrl"      : tf.keras.optimizers.legacy.Ftrl(learning_rate= learning_rate),
        "SGD"       : tf.keras.optimizers.legacy.SGD(learning_rate= learning_rate)
    }

    if optimizer not in tensorflow_optimizers.keys(): raise NotImplemented("optimizer not implemented")

    return tensorflow_optimizers[optimizer]