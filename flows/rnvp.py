import tensorflow as tf

from utils import rand_mat


class RNVPFlow:
    """Chain of `n_flows` affine flows (x = z * exp(s) + t), where (if `sample=True`)
    half of the dimensions in x are linearly scaled/transfromed as a function of the
    other half.

    Ref.: Density estimation using Real NVP, Dinh et al. May 2016
        https://arxiv.org/abs/1605.08803
    """

    def __init__(
        self,
        dim_in,
        n_flows=2,
        n_hidden=0,
        dim_h=10,
        name=None,
        nonlin=tf.tanh,
        **kwargs,
    ):
        self.dim_in = dim_in
        self.n_flows = n_flows
        self.n_hidden = n_hidden
        self.name = name
        self.dim_h = dim_h
        self.params = []
        # shape of self.params after __init__: [
        #   [[W11, b11], [W12, b12], ..., [W_mu1, b_mu1, W_sigma1, b_sigma1]],
        #   [[W21, b21], [W22, b22], ..., [W_mu2, b_mu2, W_sigma2, b_sigma2]],
        #   ...
        # ]
        # The number of weight-bias pairs [W11, b11] etc. per line is 1 + n_hidden.
        # The number of lines is n_flows.
        self.nonlin = nonlin
        for flow in range(self.n_flows):
            self._build_mnn()

    def _build_mnn(self):
        dim_in, dim_h = self.dim_in, self.dim_h
        w = rand_mat([dim_in, dim_h])
        b = tf.Variable(tf.zeros(dim_h))
        self.params.append([[w, b]])
        for l in range(self.n_hidden):
            wh = rand_mat((dim_h, dim_h))
            bh = tf.Variable(tf.zeros(dim_h))
            self.params[-1].append([wh, bh])
        w_mu = rand_mat([dim_h, dim_in])
        b_mu = tf.Variable(tf.zeros(dim_in))
        w_sigma = rand_mat([dim_h, dim_in])
        b_sigma = tf.Variable(2 * tf.ones(dim_in))
        self.params[-1].append([w_mu, b_mu, w_sigma, b_sigma])

    def feed_forward(self, x, weights):
        for j in range(len(weights[:-1])):
            h = tf.matmul(x, weights[j][0]) + weights[j][1]
            x = self.nonlin(h)
        w_mu, b_mu, w_sigma, b_sigma = weights[-1]
        mean = tf.matmul(x, w_mu) + b_mu
        sigma = tf.matmul(x, w_sigma) + b_sigma
        return mean, sigma

    def forward(self, z, sample=True):
        """Forward transformation (z -> x) applying `n_flows` RealNVP flows.
        """
        logdets = tf.zeros(z.shape[0])
        for flow in range(self.n_flows):
            # Get random Bernoulli mask. This decides which channels will remain
            # unchanged and which will be transformed as functions of the unchanged.
            mask = tf.keras.backend.random_binomial(z.shape, p=0.5) if sample else 0.5
            ggmu, ggsigma = self.feed_forward(mask * z, self.params[flow])
            gate = tf.nn.sigmoid(ggsigma)
            logdets += tf.reduce_sum((1 - mask) * tf.math.log(gate), axis=1)
            z = (1 - mask) * (z * gate + (1 - gate) * ggmu) + mask * z

        return z, logdets
