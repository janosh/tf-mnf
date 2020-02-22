import numpy as np
import tensorflow as tf

from flows import IAF, NormalizingFlow


class DenseNF(tf.keras.layers.Layer):
    """Bayesian fully-connected layer with weight posterior modeled by diagonal
    covariance Gaussian. To increase expressiveness and allow for multimodality and
    non-zero covariance between weights, the Gaussian means depend on an auxiliary
    random variable z modelled by a normalizing flow. The flows base distribution is a
    standard normal.

    From "Multiplicative Normalizing Flows for Variational Bayesian Neural Networks",
    Christos Louizos, Max Welling (Jun 2017) https://arxiv.org/abs/1703.01961
    """

    def __init__(
        self,
        n_out,
        n_flows_q=2,
        n_flows_r=2,
        learn_p=False,
        use_z=True,
        prior_var_w=1,
        prior_var_b=1,
        flow_dim_h=50,
        thres_std=1,
        std_init=1,
        **kwargs,
    ):
        self.n_out = n_out
        self.learn_p = learn_p
        self.prior_var_w = prior_var_w
        self.prior_var_b = prior_var_b
        self.thres_std = thres_std
        self.std_init = std_init
        self.n_flows_q = n_flows_q
        self.n_flows_r = n_flows_r
        self.use_z = use_z
        self.flow_dim_h = flow_dim_h
        super().__init__(**kwargs)

    def build(self, input_shape):
        n_in = self.n_in = input_shape[-1]
        n_out, flow_dim_h = self.n_out, self.flow_dim_h
        std_init, mean_init = self.std_init, -9

        glorot = tf.keras.initializers.GlorotNormal()
        self.mean_W = tf.Variable(glorot([n_in, n_out]))
        self.log_std_W = tf.Variable(glorot([n_in, n_out]) * std_init + mean_init)
        self.mean_b = tf.Variable(tf.zeros(n_out))
        self.log_var_b = tf.Variable(glorot([n_out]) * std_init + mean_init)

        if self.use_z:
            self.q0_mean = tf.Variable(
                glorot([n_in]) + (0 if self.n_flows_q > 0 else 1)
            )  # aka dropout_rates_mean
            self.q0_log_var = tf.Variable(glorot([n_in]) * std_init + mean_init)
            self.rsr_M = tf.Variable(glorot([n_in]))  # var_r_aux
            self.apvar_M = tf.Variable(glorot([n_in]))  # apvar_r_aux
            self.rsri_M = tf.Variable(glorot([n_in]))  # var_r_auxi

        self.prior_var_r_p = tf.Variable(
            glorot([n_in]) * std_init + np.log(self.prior_var_w),
            trainable=self.learn_p,
        )
        self.prior_var_r_p_bias = tf.Variable(
            glorot([1]) * std_init + np.log(self.prior_var_b), trainable=self.learn_p,
        )

        r_flows = [
            IAF(parity=i % 2, h_sizes=[flow_dim_h]) for i in range(self.n_flows_r)
        ]
        self.flow_r = NormalizingFlow(r_flows)

        q_flows = [
            IAF(parity=i % 2, h_sizes=[flow_dim_h]) for i in range(self.n_flows_q)
        ]
        self.flow_q = NormalizingFlow(q_flows)

    def sample_z(self, batch_size):
        log_dets = tf.zeros(batch_size)
        if not self.use_z:
            return tf.ones([batch_size, self.n_in]), log_dets

        q0_mean = tf.stack([self.q0_mean] * batch_size)
        epsilon = tf.random.normal([batch_size, self.n_in])
        q0_var = tf.exp(self.q0_log_var)
        z_samples = q0_mean + tf.sqrt(q0_var) * epsilon

        if self.n_flows_q > 0:
            z_samples, log_dets = self.flow_q.forward(z_samples)

        return z_samples, log_dets

    def kl_div(self):
        M, log_dets = self.sample_z(1)

        Mtilde = M[0, :, None] * self.mean_W
        Vtilde = tf.square(tf.exp(self.log_std_W))
        # outer product
        iUp = tf.stack([tf.exp(self.prior_var_r_p)] * self.n_out, axis=1)

        log_q = 0
        if self.use_z:
            # Compute entropy of the initial distribution q(z_0).
            # This is independent of the actual sample z_0.
            log_q = -0.5 * tf.reduce_sum(tf.math.log(2 * np.pi) + self.q0_log_var + 1)
            log_q -= log_dets[0]

        kldiv_w = 0.5 * tf.reduce_sum(
            tf.math.log(iUp)
            - 2 * self.log_std_W
            + (Vtilde + tf.square(Mtilde)) / iUp
            - 1
        )
        kldiv_b = 0.5 * tf.reduce_sum(
            self.prior_var_r_p_bias
            - self.log_var_b
            + (tf.exp(self.log_var_b) + tf.square(self.mean_b))
            / tf.exp(self.prior_var_r_p_bias)
            - 1
        )

        if self.use_z:
            # shared network for hidden layer
            mean_w = tf.linalg.matvec(tf.transpose(Mtilde), self.apvar_M)
            epsilon = tf.random.normal([self.n_out])
            var_w = tf.linalg.matvec(tf.transpose(Vtilde), tf.square(self.apvar_M))
            a = tf.tanh(mean_w + tf.sqrt(var_w) * epsilon)
            # split at output layer
            if len(a.shape) > 0:
                w__ = tf.reduce_mean(tf.tensordot(a, self.rsr_M, axes=0), axis=0)
                wv__ = tf.reduce_mean(tf.tensordot(a, self.rsri_M, axes=0), axis=0)
            else:
                w__ = self.rsr_M * a
                wv__ = self.rsri_M * a

            log_r = 0
            if self.n_flows_r > 0:
                M, log_r = self.flow_r.forward(M)
                log_r = log_r[0]

            log_r += 0.5 * tf.reduce_sum(
                -tf.exp(wv__) * tf.square(M - w__) - tf.math.log(2 * np.pi) + wv__
            )
        else:
            log_r = 0

        return kldiv_w + kldiv_b - log_r + log_q

    def call(self, x):
        z_samples, _ = self.sample_z(x.shape[0])
        mu_out = tf.matmul(x * z_samples, self.mean_W) + self.mean_b

        std_W = tf.clip_by_value(tf.exp(self.log_std_W), 0, self.thres_std)
        var_b = tf.clip_by_value(tf.exp(self.log_var_b), 0, self.thres_std ** 2)
        var_W = tf.square(std_W)
        V_h = tf.matmul(tf.square(x), var_W) + var_b
        epsilon = tf.random.normal(mu_out.shape)
        sigma_out = tf.sqrt(V_h) * epsilon

        return mu_out + sigma_out
