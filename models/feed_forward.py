import tensorflow as tf

from layers import DenseNF

tfkl = tf.keras.layers


class NFFeedForward(tf.keras.Sequential):
    def __init__(self, layer_dims=(100, 50, 10), activation=tfkl.ReLU, **kwargs):
        layers, batch_norm = [], tfkl.BatchNormalization
        for dim in layer_dims:
            layers.extend([DenseNF(dim, **kwargs), activation(), batch_norm()])
        super().__init__(layers)

    def kl_div(self):
        """Compute current KL divergence of the whole model.
        Can be used as a regularization term during training.
        """
        return sum([l.kl_div() for l in self.layers if hasattr(l, "kl_div")])
