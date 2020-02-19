import tensorflow as tf


class PlanarFlow(tf.Module):
    """Planar flow modifies the base density by applying a series of contractions and
    expansions in the direction perpendicular to the hyperplane w^T * z + b = 0.

    From "Variational Inference with Normalizing Flows", Rezende & Mohamed (Jun 2015)
    https://arxiv.org/abs/1505.05770
    """

    def __init__(self, dim, **kwargs):
        super().__init__(**kwargs)
        glorot = tf.keras.initializers.GlorotNormal()
        w = tf.Variable(glorot([dim, 1]))
        u = tf.Variable(glorot([dim, 1]))
        b = tf.Variable(tf.zeros(1))
        self.params = [w, u, b]

    def forward(self, z):
        w, u, b = self.params

        uw = tf.reduce_sum(u * w)
        muw = -1 + tf.math.softplus(uw)  # = -1 + log(1 + exp(uw))
        u_hat = u + (muw - uw) * w / tf.reduce_sum(w ** 2)
        if len(z.shape) == 1:
            zwb = z * w + b
        else:
            zwb = tf.matmul(z, w) + b

        # We choose tf.tanh as non-linearity.
        z_shift = tf.matmul(tf.tanh(zwb), tf.transpose(u_hat))
        z += z_shift

        # d tanh(x)/dx = 1 - tf.tanh(x)^2
        psi = tf.matmul(1 - tf.tanh(zwb) ** 2, tf.transpose(w))
        psi_u = tf.matmul(psi, u_hat)
        logdet = tf.squeeze(tf.math.log(tf.abs(1 + psi_u)))
        return z, logdet
