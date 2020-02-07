import tensorflow as tf

from utils import rand_mat


class PlanarFlow:
    """
    """

    def __init__(self, dim_in, n_flows=2, name=None, **kwargs):
        self.dim_in = dim_in
        self.n_flows = n_flows
        self.params = []
        self.name = name
        self.build()

    def build(self):
        for flow in range(self.n_flows):
            w = rand_mat([self.dim_in, 1], name=f"w_{flow}_{self.name}")
            u = rand_mat([self.dim_in, 1], name=f"u_{flow}_{self.name}")
            b = tf.Variable(tf.zeros(1), name=f"b_{flow}_{self.name}")
            self.params.append([w, u, b])

    def forward(self, z):
        logdets = tf.zeros(z.shape[0])
        for flow in range(self.n_flows):
            w, u, b = self.params[flow]
            uw = tf.reduce_sum(u * w)
            muw = -1 + tf.math.softplus(uw)  # = -1 + log(1 + exp(uw))
            u_hat = u + (muw - uw) * w / tf.reduce_sum(w ** 2)
            if len(z.shape) == 1:
                zwb = z * w + b
            else:
                zwb = tf.matmul(z, w) + b
            psi = tf.matmul(1 - tf.tanh(zwb) ** 2, tf.transpose(w))
            psi_u = tf.matmul(psi, u_hat)
            logdets += tf.squeeze(tf.math.log(tf.abs(1 + psi_u)))
            zadd = tf.matmul(tf.tanh(zwb), tf.transpose(u_hat))
            z += zadd
        return z, logdets
