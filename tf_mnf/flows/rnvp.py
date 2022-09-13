import tensorflow as tf


class RNVP(tf.Module):
    """Affine half (aka Real Non-Volume Preserving) flow (x = z * exp(s) + t),
    where a randomly selected half z1 of the dimensions in z are transformed as an
    affine function of the other half z2, i.e. scaled by s(z2) and shifted by t(z2).

    From "Density estimation using Real NVP", Dinh et al. (May 2016)
    https://arxiv.org/abs/1605.08803

    This implementation uses the numerically stable updates introduced by IAF:
    https://arxiv.org/abs/1606.04934
    """

    def __init__(self, dim, h_sizes=(30,), activation="tanh", **kwargs):
        super().__init__(**kwargs)
        layers = [tf.keras.layers.Dense(hs, activation) for hs in h_sizes]
        self.net = tf.keras.Sequential(layers)
        self.t = tf.keras.layers.Dense(dim)
        self.s = tf.keras.layers.Dense(dim)

    def forward(self, z):  # z -> x
        # Get random Bernoulli mask. This decides which channels will remain
        # unchanged and which will be transformed as functions of the unchanged.
        mask = tf.keras.backend.random_binomial(tf.shape(z), p=0.5)
        z1, z2 = (1 - mask) * z, mask * z
        y = self.net(z2)
        shift = self.t(y)
        scale = self.s(y)

        # sigmoid(x) = 1 / (1 + exp(-x)). For x in (-inf, inf) => sigmoid(x) in (0, 1).
        gate = tf.sigmoid(scale)
        log_dets = tf.reduce_sum((1 - mask) * tf.math.log(gate), axis=1)
        x = (z1 * gate + (1 - gate) * shift) + z2

        return x, log_dets
