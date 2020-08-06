import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, ReLU

from ..layers import DenseMNF


class MNFFeedForward(tf.keras.Sequential):
    def __init__(self, layer_sizes=(100, 50, 10), activation=ReLU, **kwargs):
        layers = []
        for dim in layer_sizes[:-1]:
            layers.extend([DenseMNF(dim, **kwargs), activation(), BatchNormalization()])
        super().__init__(layers + [DenseMNF(layer_sizes[-1], **kwargs)])

    def kl_div(self):
        """Compute current KL divergence of the whole model.
        Can be used as a regularization term during training.
        """
        return sum([lyr.kl_div() for lyr in self.layers if hasattr(lyr, "kl_div")])
