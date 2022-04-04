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
        self.u = tf.Variable(glorot([dim, 1]))
        self.dense = tf.keras.layers.Dense(1)

    def forward(self, z):  # z -> x
        w = self.dense.kernel
        uw = tf.reduce_sum(self.u * w)
        suw = -1 + tf.math.softplus(uw)  # = -1 + log(1 + exp(uw))
        u_hat = self.u + (suw - uw) * w / tf.reduce_sum(w**2)

        zwb = self.dense(z)
        shift = tf.matmul(tf.tanh(zwb), tf.transpose(u_hat))
        x = z + shift

        # d tanh(x)/dx = 1 - tf.tanh(x)^2
        psi = tf.matmul(1 - tf.tanh(zwb) ** 2, tf.transpose(w))
        psi_u = tf.matmul(psi, u_hat)
        logdet = tf.squeeze(tf.math.log(tf.abs(1 + psi_u)))
        return x, logdet
