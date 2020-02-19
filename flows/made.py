"""
Implements a Masked Autoregressive MLP, where carefully constructed
binary masks over weights ensure the autoregressive property.
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


class MADE(tf.keras.Sequential):
    """Masked Autoencoder for Distribution Estimation masks the autoencoder’s
    parameters to respect autoregressive constraints: each input is reconstructed
    only from previous inputs in a given ordering. Constrained this way, the
    autoencoder outputs can be interpreted as a set of conditional probabilities and
    their product as the full joint probability. We can also train a single network
    that can decompose the joint probability in multiple different orderings.

    Germain et al. (June 2015) https://arxiv.org/abs/1502.03509
    """

    def __init__(self, n_in, n_out, h_sizes=[], num_masks=1, shuffle=True, **kwargs):
        """
        n_in (int): number of inputs
        h_sizes (list of ints): number of units in hidden layers
        n_out (int): number of outputs, which usually collectively parameterize some
            kind of 1D distribution
            note: if n_out is e.g. 2x larger than n_in (perhaps the mean and std), then
            the first n_in will be all the means and the second n_in will be stds. i.e.
            output dimensions depend on the same input dimensions in "chunks" and should
            be carefully decoded downstream appropriately. the output of runn_ing the
            tests for this file makes this a bit more clear with examples.
        num_masks: can be used to train ensemble over orderings/connections
        shuffle: Whether to apply random permutations to ordering of the inputs.
        """
        assert n_out % n_in == 0, "n_out must be integer multiple of n_in"
        self.n_outputs = n_out // n_in  # Integer division to avoid type errors.
        self.n_in = n_in
        self.h_sizes = h_sizes

        # define a simple MLP neural net
        layers = [tf.keras.Input(n_in)]
        for size in h_sizes + [n_out]:
            layers.extend([MaskedDense(size), tf.keras.layers.ReLU()])
        layers.pop()  # pop the last ReLU for the output layer
        super().__init__(layers, **kwargs)

        # Seeds for orders/connectivities of the model ensemble.
        self.shuffle = shuffle
        self.num_masks = num_masks
        self.seed = 0  # for cycling through num_masks orderings

        self.m = {}
        self.set_masks()  # Build the initial self.m connectivity
        # Note: We could also precompute the masks and cache them, but this
        # could get memory expensive for large numbers of masks.

    def call(self, x):
        return tf.split(super().call(x), self.n_outputs, axis=-1)

    def set_masks(self):
        if self.m and self.num_masks == 1:
            return  # Only a single seed, skip for efficiency.
        n_layers = len(self.h_sizes)

        # Fetch the next seed and construct a random stream.
        rng = np.random.RandomState(self.seed)
        self.seed = (self.seed + 1) % self.num_masks

        # Sample the inputs order and the connectivity of all neurons.
        self.m[-1] = (
            rng.permutation(self.n_in) if self.shuffle else np.arange(self.n_in)
        )
        for l in range(n_layers):
            # Use minimum connectivity of previous layer as lower bound when sampling
            # values for m_l(k) to avoid unconnected units. See comment after eq. (13).
            self.m[l] = rng.randint(
                self.m[l - 1].min(), self.n_in - 1, size=self.h_sizes[l]
            )

        # Construct the mask matrices.
        masks = [self.m[l - 1][:, None] <= self.m[l][None, :] for l in range(n_layers)]
        masks.append(self.m[n_layers - 1][:, None] < self.m[-1][None, :])

        # Handle the case where n_out = n_in * k, for integer k > 1.
        if self.n_outputs > 1:
            # Replicate the mask across all outputs.
            masks[-1] = np.concatenate([masks[-1]] * self.n_outputs, axis=1)

        # Update the masks in all MaskedDense layers.
        layers = [l for l in self.layers if isinstance(l, MaskedDense)]
        for l, m in zip(layers, masks):
            l.set_mask(m)
