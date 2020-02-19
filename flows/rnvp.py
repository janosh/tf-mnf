import tensorflow as tf


class RNVP(tf.Module):
    """Affine half (aka real non-volume preserving) flow (x = z * exp(s) + t),
    where a randomly selected half of the dimensions in x are linearly
    scaled/transfromed as a function of the other half.

    From "Density estimation using Real NVP", Dinh et al. (May 2016).
    https://arxiv.org/abs/1605.08803
    """

    def __init__(self, dim, h_layers=0, dim_h=10, nonlin=tf.tanh, **kwargs):
        super().__init__(**kwargs)
        self.params = []
        # shape of self.params: [W1, b1], [W2, b2], ..., [W_mu, b_mu, W_sigma, b_sigma]
        # The number of weight-bias pairs [W1, b1], ... is 1 + h_layers.
        self.nonlin = nonlin

        glorot = tf.keras.initializers.GlorotNormal()
        w = tf.Variable(glorot([dim, dim_h]))
        b = tf.Variable(tf.zeros(dim_h))
        self.params.append([w, b])
        for l in range(h_layers):
            wh = tf.Variable(glorot([dim_h, dim_h]))
            bh = tf.Variable(tf.zeros(dim_h))
            self.params.append([wh, bh])
        w_mu = tf.Variable(glorot([dim_h, dim]))
        b_mu = tf.Variable(tf.zeros(dim))
        w_sigma = tf.Variable(glorot([dim_h, dim]))
        b_sigma = tf.Variable(tf.ones(dim))
        self.params.append([w_mu, b_mu, w_sigma, b_sigma])

    def feed_forward(self, x):
        for [w, b] in self.params[:-1]:
            h = tf.matmul(x, w) + b
            x = self.nonlin(h)
        w_mu, b_mu, w_sigma, b_sigma = self.params[-1]
        mu = tf.matmul(x, w_mu) + b_mu
        sigma = tf.matmul(x, w_sigma) + b_sigma
        return mu, sigma

    def forward(self, z):  # z -> x
        # Get random Bernoulli mask. This decides which channels will remain
        # unchanged and which will be transformed as functions of the unchanged.
        mask = tf.keras.backend.random_binomial(z.shape, p=0.5)

        mu, sigma = self.feed_forward(mask * z)
        gate = tf.nn.sigmoid(sigma)
        log_dets = tf.reduce_sum((1 - mask) * tf.math.log(gate), axis=1)
        z = (1 - mask) * (z * gate + (1 - gate) * mu) + mask * z

        return z, log_dets
