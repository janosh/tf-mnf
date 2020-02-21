"""Adapted from https://github.com/karpathy/pytorch-normalizing-flows."""

import tensorflow as tf

from flows.made import MADE


class MAF(tf.Module):
    """Masked Autoregressive Flow that uses a MADE-style network for fast single-pass
    inverse() (for density estimation) but slow dim-times-pass forward() (for sampling).

    From "Masked Autoregressive Flow for Density Estimation", Papamakarios et al.
    (Jun 2018) https://arxiv.org/abs/1705.07057
    """

    def __init__(self, parity, net=None, h_sizes=[24]):
        super().__init__()
        self.parity = parity
        # Uses a 2-layer auto-regressive MLP with 2 outputs by default.
        # Custom nets must also have 2 outputs, one for log scale s and one for shift t.
        self.net = net or MADE(n_outputs=2, h_sizes=h_sizes)

    def forward(self, z):
        batch_size, dim = z.shape  # dim: the flow's input and output dimensionality.
        x = tf.zeros_like(z)
        log_dets = tf.zeros(batch_size)
        # Reverse order, so that if we chain MAFs, we spread expressivity equally
        # over all dimensions.
        z = tf.reverse(z, axis=[1]) if self.parity else z
        # x has to be decoded sequentially, one element at a time.
        for i in range(dim):
            s, t = self.net(x)
            x[:, i] = (z[:, i] - t[:, i]) * tf.exp(-s[:, i])
            log_dets += -s[:, i]
        return x, log_dets

    def inverse(self, x):
        # Since we can evaluate all of z in parallel, density estimation is fast.
        s, t = self.net(x)
        z = x * tf.exp(s) + t
        z = tf.reverse(z, axis=[1]) if self.parity else z
        log_dets = tf.reduce_sum(s, axis=1)
        return z, log_dets


class IAF(MAF):
    """Reverses the flow of MAF, giving an Inverse Autoregressive Flow (IAF)
    which offers fast sampling but slow density estimation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.forward, self.inverse = self.inverse, self.forward
