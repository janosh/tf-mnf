import tensorflow as tf

from flows.maf import IAF, MAF  # noqa
from flows.planar import PlanarFlow  # noqa
from flows.rnvp import RNVP  # noqa


class NormalizingFlow(tf.Module):
    """Diffeomorphisms are a group. Hence a sequence of normalizing flows
    is itself a normalizing flow.
    """

    def __init__(self, flows, **kwargs):
        super().__init__(**kwargs)
        self.flows = flows

    def forward(self, z, direction="forward", with_steps=False):  # z -> x
        log_dets = tf.zeros(z.shape[0])
        xs = [z]
        for flow in reversed(self.flows) if direction == "inverse" else self.flows:
            z, ld = getattr(flow, direction)(z)
            log_dets += ld
            xs.append(z)
        return xs if with_steps else xs[-1], log_dets

    def inverse(self, x):  # x -> z
        return self.forward(x, direction="inverse")


class NormalizingFlowModel(NormalizingFlow):
    """A normalizing flow model is a (base distro, flow) pair."""

    def __init__(self, base, flows):
        super().__init__(flows)
        # Distribution class that exposes a log_prob() and sample() method.
        self.base = base

    def base_log_prob(self, x):
        z, _ = self.inverse(x)
        return tf.reduce_sum(self.base.log_prob(z), axis=1)

    def sample(self, *num_samples, **kwargs):
        z = self.base.sample(num_samples)
        # Pass with_steps=True if you need both z and x.
        x, _ = self.forward(z, **kwargs)
        return x
