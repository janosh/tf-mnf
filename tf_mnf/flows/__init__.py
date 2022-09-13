import tensorflow as tf

from tf_mnf.flows.maf import IAF as IAF
from tf_mnf.flows.maf import MAF as MAF
from tf_mnf.flows.maf import TFIAF as TFIAF
from tf_mnf.flows.maf import TFMAF as TFMAF
from tf_mnf.flows.planar import PlanarFlow as PlanarFlow
from tf_mnf.flows.rnvp import RNVP as RNVP


class NormalizingFlow(tf.Module):
    """Diffeomorphisms are a group. Hence a sequence of normalizing flows
    is itself a normalizing flow.
    """

    def __init__(self, flows):
        super().__init__()
        self.flows = flows

    def forward(self, z):  # z -> x
        log_dets = tf.zeros(tf.shape(z)[0])
        xs = [z]  # ensure z is tensor rathen than array
        for flow in self.flows:
            z, ld = flow.forward(z)
            log_dets += ld
            xs.append(z)
        return xs, log_dets

    def inverse(self, x):  # x -> z
        log_dets = tf.zeros(tf.shape(x)[0])
        zs = [x]
        for flow in reversed(self.flows):
            x, ld = flow.inverse(x)
            log_dets += ld
            zs.append(x)
        return zs, log_dets


class NormalizingFlowModel(NormalizingFlow):
    """A normalizing flow model is a (base distro, flow) pair."""

    def __init__(self, base, flow):
        super().__init__(flow)
        # Distribution class that exposes a log_prob() and sample() method.
        self.base = base

    def base_log_prob(self, x):
        zs, _ = self.inverse(x)
        return tf.reduce_sum(self.base.log_prob(zs[-1]))

    def sample(self, *num_samples):
        z = self.base.sample(num_samples)
        # Pass with_steps=True if you need both z and x.
        xs, _ = self.forward(z)
        return xs
