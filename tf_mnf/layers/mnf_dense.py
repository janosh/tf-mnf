import numpy as np
import tensorflow as tf

from tf_mnf.flows import IAF, NormalizingFlow


class MNFDense(tf.keras.layers.Layer):
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
        flow_h_sizes=(50,),
        max_std=1,
        std_init=1,
        **kwargs,
    ):
        self.n_out = n_out
        self.learn_p = learn_p
        self.prior_var_w = prior_var_w
        self.prior_var_b = prior_var_b
        self.max_std = max_std
        self.std_init = std_init
        self.n_flows_q = n_flows_q
        self.n_flows_r = n_flows_r
        self.use_z = use_z
        self.flow_h_sizes = flow_h_sizes
        super().__init__(**kwargs)

    def build(self, input_shape):
        n_in = self.n_in = input_shape[-1]
        std_init, mean_init = self.std_init, -9

        glorot = tf.keras.initializers.GlorotNormal()
        self.mean_W = tf.Variable(glorot([n_in, self.n_out]))
        self.log_std_W = tf.Variable(glorot([n_in, self.n_out]) * std_init + mean_init)
        self.mean_b = tf.Variable(tf.zeros(self.n_out))
        self.log_var_b = tf.Variable(glorot([self.n_out]) * std_init + mean_init)

        if self.use_z:
            # q0_mean has similar function to a dropout rate as it determines the
            # mean of the multiplicative noise z_i in eq. 4.
            self.q0_mean = tf.Variable(
                glorot([n_in]) + (0 if self.n_flows_q > 0 else 1)
            )
            self.q0_log_var = tf.Variable(glorot([n_in]) * std_init + mean_init)

            self.r0_mean = tf.Variable(glorot([n_in]))
            self.r0_log_var = tf.Variable(glorot([n_in]))
            self.r0_apvar = tf.Variable(glorot([n_in]))

        self.prior_var_r_p = tf.Variable(
            glorot([n_in]) * std_init + np.log(self.prior_var_w),
            trainable=self.learn_p,
        )
        self.prior_var_r_p_bias = tf.Variable(
            glorot([1]) * std_init + np.log(self.prior_var_b), trainable=self.learn_p
        )

        r_flows = [
            IAF(parity=i % 2, h_sizes=self.flow_h_sizes) for i in range(self.n_flows_r)
        ]
        self.flow_r = NormalizingFlow(r_flows)

        q_flows = [
            IAF(parity=i % 2, h_sizes=self.flow_h_sizes) for i in range(self.n_flows_q)
        ]
        self.flow_q = NormalizingFlow(q_flows)

    def sample_z(self, batch_size):
        log_dets = tf.zeros(batch_size)
        if not self.use_z:
            return tf.ones([batch_size, self.n_in]), log_dets

        q0_mean = tf.tile(self.q0_mean[None, :], [batch_size, 1])
        epsilon = tf.random.normal([batch_size, self.n_in])
        q0_var = tf.exp(self.q0_log_var)
        z_samples = q0_mean + tf.sqrt(q0_var) * epsilon

        if self.n_flows_q > 0:
            z_samples, log_dets = self.flow_q.forward(z_samples)
            z_samples = z_samples[-1]

        return z_samples, log_dets

    def kl_div(self):
        z_sample, log_det_q = self.sample_z(1)

        Mtilde = tf.transpose(z_sample) * self.mean_W
        Vtilde = tf.square(tf.exp(self.log_std_W))
        # Stacking yields same result as outer product with ones. See eqs. 9, 10.
        iUp = tf.stack([tf.exp(self.prior_var_r_p)] * self.n_out, axis=1)

        kl_div_w = 0.5 * tf.reduce_sum(
            tf.math.log(iUp)
            - 2 * self.log_std_W
            + (Vtilde + tf.square(Mtilde)) / iUp
            - 1
        )
        kl_div_b = 0.5 * tf.reduce_sum(
            self.prior_var_r_p_bias
            - self.log_var_b
            + (tf.exp(self.log_var_b) + tf.square(self.mean_b))
            / tf.exp(self.prior_var_r_p_bias)
            - 1
        )

        log_q = -tf.squeeze(log_det_q)
        if self.use_z:
            # Compute entropy of the initial distribution q(z_0).
            # This is independent of the actual sample z_0.
            log_q -= 0.5 * tf.reduce_sum(tf.math.log(2 * np.pi) + self.q0_log_var + 1)

        log_r = 0
        if self.use_z:
            if self.n_flows_r > 0:
                z_sample, log_det_r = self.flow_r.forward(z_sample)
                log_r = tf.squeeze(log_det_r)

            # Shared network for hidden layer.
            mean_w = tf.linalg.matvec(tf.transpose(Mtilde), self.r0_apvar)
            var_w = tf.linalg.matvec(tf.transpose(Vtilde), tf.square(self.r0_apvar))
            epsilon = tf.random.normal([self.n_out])
            # The bias contribution is not included in `a` since the multiplicative
            # noise is at the input units (hence it doesn't affect the biases)
            a = tf.tanh(mean_w + tf.sqrt(var_w) * epsilon)
            # Split at output layer. Use tf.tensordot for outer product.
            mean_r = tf.reduce_mean(tf.tensordot(a, self.r0_mean, axes=0), axis=0)
            log_var_r = tf.reduce_mean(tf.tensordot(a, self.r0_log_var, axes=0), axis=0)
            # mu_tilde & sigma_tilde from eqs. 9, 10: mean and log var of the auxiliary
            # normal dist. r(z_T_b|W) from eq. 8. Used to compute first term in 15.

            log_r += 0.5 * tf.reduce_sum(
                -tf.exp(log_var_r) * tf.square(z_sample - mean_r)
                - tf.math.log(2 * np.pi)
                + log_var_r
            )

        return kl_div_w + kl_div_b - log_r + log_q

    def call(self, x):
        z_samples, _ = self.sample_z(tf.shape(x)[0])
        mu_out = tf.matmul(x * z_samples, self.mean_W) + self.mean_b

        std_W = tf.clip_by_value(tf.exp(self.log_std_W), 0, self.max_std)
        var_b = tf.clip_by_value(tf.exp(self.log_var_b), 0, self.max_std**2)
        var_W = tf.square(std_W)
        V_h = tf.matmul(tf.square(x), var_W) + var_b
        epsilon = tf.random.normal(tf.shape(mu_out))
        sigma_out = tf.sqrt(V_h) * epsilon

        return mu_out + sigma_out
