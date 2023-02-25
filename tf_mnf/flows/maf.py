"""Adapted from https://github.com/karpathy/pytorch-normalizing-flows."""

import tensorflow as tf
import tensorflow_probability as tfp

from tf_mnf.flows.made import MADE


class MAF(tf.Module):
    """Masked Autoregressive Flow that uses a MADE-style network for fast single-pass
    inverse() (for density estimation) but slow dim-times-pass forward() (for sampling).

    From "Masked Autoregressive Flow for Density Estimation", Papamakarios et al.
    (Jun 2018) https://arxiv.org/abs/1705.07057
    """

    def __init__(self, parity, net=None, h_sizes=(30,)):
        super().__init__()
        self.parity = parity
        # Uses a 2-layer auto-regressive MLP with 2 outputs by default.
        # Custom nets must also have 2 outputs, one for log scale s and one for shift t.
        self.net = net or MADE(n_outputs=2, h_sizes=h_sizes)

    def forward(self, z):
        batch_size, dim = tf.shape(z)  # dim: the flow's dimensionality.
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


class TFMAF(tfp.bijectors.MaskedAutoregressiveFlow):
    """Wrapper around TFP's MaskedAutoregressiveFlow."""

    def __init__(self, made=None, h_sizes=(30, 30), **kwargs):
        """Create a Masked Autoregressive Flow (MAF) bijector.

        Args:
            made (keras.Model): Masked autoencoder to use as shift and log scale func.
            h_sizes (list[int]): size of hidden layers in the MADE network. Ignored if
                made is not None.
            **kwargs: Passed to tfp.bijectors.MaskedAutoregressiveFlow.
        """
        if not made:  # Define a default masked autoencoder for density estimation.
            made = tfp.bijectors.AutoregressiveNetwork(params=2, hidden_units=h_sizes)
        super().__init__(shift_and_log_scale_fn=made, **kwargs)

    def forward(self, x):
        # 2nd arg (event_ndims) to the Jacobian computation indicates the Number of
        # dimensions in the probabilistic events being transformed. Just 1 here
        # because every sample in x is assumed independent.
        return super().forward(x), super().forward_log_det_jacobian(x, 1)

    def inverse(self, x):
        return super().inverse(x), super().inverse_log_det_jacobian(x, 1)


class TFIAF(TFMAF):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.forward, self.inverse = self.inverse, self.forward
