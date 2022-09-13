import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, ReLU

from tf_mnf.layers import MNFDense


class MNFFeedForward(tf.keras.Sequential):
    def __init__(self, layer_sizes=(100, 50, 10), activation=ReLU, **kwargs):
        layers = []
        for dim in layer_sizes[:-1]:
            layers.extend([MNFDense(dim, **kwargs), activation(), BatchNormalization()])
        super().__init__(layers + [MNFDense(layer_sizes[-1], **kwargs)])

    def kl_div(self):
        """Compute current KL divergence of the whole model. Should be included
        as a regularization term in the loss function. Tensorflow will issue
        warnings "Gradients do not exist for variables of MNFDense" if you forget.
        """
        return sum(lyr.kl_div() for lyr in self.layers if hasattr(lyr, "kl_div"))
