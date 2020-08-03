import tensorflow as tf
from layers import Conv2DNF, DenseNF
from tensorflow.keras.layers import Flatten, MaxPool2D, ReLU, Softmax


class NFLeNet(tf.keras.Sequential):
    """Bayesian LeNet with parameter posteriors modeled by normalizing flows."""

    def __init__(self, layer_dims=[20, 50, 500, 10], **kwargs):
        c1 = Conv2DNF(layer_dims[0], 5, padding="VALID", **kwargs)
        r1 = ReLU()
        mp1 = MaxPool2D(padding="SAME")
        c2 = Conv2DNF(layer_dims[1], 5, padding="VALID", **kwargs)
        r2 = ReLU()
        mp2 = MaxPool2D(padding="SAME")
        f = Flatten()
        d1 = DenseNF(layer_dims[2], **kwargs)
        r3 = ReLU()
        d2 = DenseNF(layer_dims[3], **kwargs)
        s = Softmax()
        super().__init__([c1, r1, mp1, c2, r2, mp2, f, d1, r3, d2, s])

    def kl_div(self):
        """Compute current KL divergence of the whole model.
        Can be used as a regularization term during training.
        """
        return sum([lyr.kl_div() for lyr in self.layers if hasattr(lyr, "kl_div")])
