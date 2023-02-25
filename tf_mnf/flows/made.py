"""Implements a Masked Autoregressive Density Estimator, where carefully
constructed binary masks over weights ensure autoregressivity.
Adapted from https://github.com/karpathy/pytorch-made.
"""

import numpy as np
import tensorflow as tf


class MaskedDense(tf.keras.layers.Dense):
    """A dense layer with a configurable mask on the weights."""

    def __init__(self, n_out, **kwargs):
        super().__init__(n_out, **kwargs)

    def build(self, input_shape):
        super().build(input_shape)
        n_in = input_shape[1]
        self.mask = tf.ones([n_in, self.units])

    def set_mask(self, mask):
        self.mask = tf.cast(tf.cast(mask, bool), float)

    def call(self, x):
        w, b = self.weights
        return tf.matmul(x, self.mask * w) + b


class MADE(tf.keras.layers.Layer):
    """Masked Autoencoder for Distribution Estimation masks the autoencoder’s
    parameters to respect autoregressive constraints: each output is reconstructed
    only from previous inputs in a given ordering. Constrained this way, the
    autoencoder outputs can be interpreted as a set of conditional probabilities and
    their product as the full joint probability. We can also train a single network
    that can decompose the joint probability in multiple different orderings.

    Germain et al. (June 2015) https://arxiv.org/abs/1502.03509
    """

    def __init__(self, n_outputs=1, h_sizes=(), num_masks=1, shuffle=False, **kwargs):
        """Create a Masked Autoencoder for Distribution Estimation.

        Args:
            n_in (int): number of inputs
            h_sizes (list[int]): number of units in hidden layers
            n_outputs (int): number of outputs, which usually collectively parameterize
                some kind of 1D distribution
                note: if n_out is e.g. 2x larger than n_in (perhaps the mean and std),
                then the first n_in will be all the means and the second n_in will be
                stds. i.e. output dimensions depend on the same input dimensions in
                "chunks" and should be carefully decoded downstream appropriately. The
                output of running the tests for this file makes this a bit more clear
                with examples.
            num_masks (int): can be used to train ensemble over orderings/connections
            shuffle (bool): Whether to apply a random permutation to the input ordering.
            **kwargs: Passed to tf.keras.layers.Layer.
        """
        super().__init__(**kwargs)
        self.n_outputs = n_outputs
        self.h_sizes = h_sizes
        # Seeds for orders/connectivities of the model ensemble.
        self.shuffle = shuffle
        self.num_masks = num_masks
        self.seed = 0  # For cycling through num_masks orderings.

    def build(self, input_shape):
        self.n_in = input_shape[-1]
        # Simple feed-forward net built with masked layers to make it autoregressive.
        layers = []
        for size in [*self.h_sizes, self.n_in * self.n_outputs]:
            layers.extend([MaskedDense(size), tf.keras.layers.ReLU()])
        self.net = tf.keras.Sequential(layers[:-1])  # drop last ReLU

        self.m = {}
        self.set_masks()  # Build the initial self.m connectivity
        # Note: We could also precompute the masks and cache them, but this
        # could get memory expensive for large numbers of masks.

    def call(self, x):
        return tf.split(self.net(x), self.n_outputs, axis=-1)

    def set_masks(self):
        if self.m and self.num_masks == 1:
            return  # Only a single seed, skip for efficiency.

        # Construct a random number generator and fetch the next seed.
        rng = np.random.RandomState(self.seed)
        self.seed = (self.seed + 1) % self.num_masks

        # Sample the inputs order and the connectivity of all neurons.
        self.m[-1] = (
            rng.permutation(self.n_in) if self.shuffle else np.arange(self.n_in)
        )
        for lyr, size in enumerate(self.h_sizes):
            # Use minimum connectivity of previous layer as lower bound when sampling
            # values for m_l(k) to avoid unconnected units. See comment after eq. (13).
            self.m[lyr] = rng.randint(self.m[lyr - 1].min(), self.n_in - 1, size=size)

        # Construct the mask matrices.
        n_layers = len(self.h_sizes)
        masks = [
            self.m[lyr - 1][:, None] <= self.m[lyr][None, :] for lyr in range(n_layers)
        ]
        masks.append(self.m[n_layers - 1][:, None] < self.m[-1][None, :])

        if self.n_outputs > 1:
            # In case of multiple outputs for each input, replicate the final layer's
            # mask across all outputs.
            masks[-1] = np.concatenate([masks[-1]] * self.n_outputs, axis=1)

        # Update the masks in all MaskedDense layers.
        layers = [lyr for lyr in self.net.layers if isinstance(lyr, MaskedDense)]
        for lyr, m in zip(layers, masks):
            lyr.set_mask(m)
