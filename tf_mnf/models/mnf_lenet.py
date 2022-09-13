import tensorflow as tf
from tensorflow.keras.layers import Flatten, MaxPool2D, ReLU, Softmax

from tf_mnf.layers import MNFConv2D, MNFDense


class MNFLeNet(tf.keras.Sequential):
    """Bayesian LeNet with parameter posteriors modeled by normalizing flows."""

    def __init__(self, s1=20, s2=50, s3=500, s4=10, **kwargs):
        c1 = MNFConv2D(s1, 5, padding="VALID", **kwargs)
        r1 = ReLU()
        mp1 = MaxPool2D(padding="SAME")
        c2 = MNFConv2D(s2, 5, padding="VALID", **kwargs)
        r2 = ReLU()
        mp2 = MaxPool2D(padding="SAME")
        f = Flatten()
        d1 = MNFDense(s3, **kwargs)
        r3 = ReLU()
        d2 = MNFDense(s4, **kwargs)
        s = Softmax()
        super().__init__([c1, r1, mp1, c2, r2, mp2, f, d1, r3, d2, s])

    def kl_div(self):
        """Compute current KL divergence of the whole model.
        Can be used as a regularization term during training.
        """
        return sum(lyr.kl_div() for lyr in self.layers if hasattr(lyr, "kl_div"))
