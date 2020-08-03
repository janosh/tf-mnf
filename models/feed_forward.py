import tensorflow as tf
from layers import DenseNF
from tensorflow.keras.layers import BatchNormalization, ReLU


class NFFeedForward(tf.keras.Sequential):
    def __init__(self, layer_dims=(100, 50, 10), activation=ReLU, **kwargs):
        layers, batch_norm = [], BatchNormalization
        for dim in layer_dims:
            layers.extend([DenseNF(dim, **kwargs), activation(), batch_norm()])
        super().__init__(layers)

    def kl_div(self):
        """Compute current KL divergence of the whole model.
        Can be used as a regularization term during training.
        """
        return sum([lyr.kl_div() for lyr in self.layers if hasattr(lyr, "kl_div")])
