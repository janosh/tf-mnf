import numpy as np
import tensorflow as tf

from flows import IAF, NormalizingFlow


class Conv2DNF(tf.keras.layers.Layer):
    """Bayesian 2D convolutional layer with weight posterior modeled by diagonal
    covariance Gaussian. To increase expressiveness and allow for multimodality and
    non-zero covariance between weights, the Gaussian means depend on an auxiliary
    random variable z modelled by a normalizing flow. The flows base distribution is a
    standard normal.

    From "Multiplicative Normalizing Flows for Variational Bayesian Neural Networks",
    Christos Louizos, Max Welling (Jun 2017) https://arxiv.org/abs/1703.01961
    """

    def __init__(
        self,
        n_filters,  # int: Dimensionality of the output space.
        kernel_size,  # int or list of two ints for kernel height and width.
        # Stride of the sliding kernel for each of the 4 input dimension
        # (batch_size, vertical, horizontal, stack_size).
        strides=1,  # int or list of ints of length 1, 2 or 4.
        padding="SAME",  # "SAME" or "VALID"
        n_flows_q=2,
        n_flows_r=2,
        learn_p=False,
        use_z=True,
        prior_var_w=1,
        prior_var_b=1,
        flow_h_sizes=[50],
        max_std=1,
        std_init=1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_filters = n_filters
        self.kernel_size = (
            [kernel_size, kernel_size] if type(kernel_size) == int else kernel_size
        )
        self.padding = padding
        self.strides = strides
        self.max_std = max_std
        self.std_init = std_init
        self.flow_h_sizes = flow_h_sizes
        self.learn_p = learn_p
        self.prior_var_w = prior_var_w
        self.prior_var_b = prior_var_b
        self.n_flows_q = n_flows_q
        self.n_flows_r = n_flows_r
        self.use_z = use_z

    def build(self, input_shape):
        stack_size = input_shape[-1]  # = 1 for black & white images like MNIST
        std_init, mean_init = self.std_init, -9
        n_rows, n_cols = self.kernel_size
        n_filters = self.n_filters
        self.input_dim = n_cols * stack_size * n_rows
        W_shape = (n_rows, n_cols, stack_size, n_filters)

        glorot = tf.keras.initializers.GlorotNormal()
        self.mean_W = tf.Variable(glorot(W_shape))
        self.log_std_W = tf.Variable(glorot(W_shape) * std_init + mean_init)
        self.mean_b = tf.Variable(tf.zeros(n_filters))
        self.log_var_bias = tf.Variable(glorot([n_filters]) * std_init + mean_init)

        if self.use_z:
            self.q0_mean = tf.Variable(
                glorot([n_filters]) + (0 if self.n_flows_q > 0 else 1)
            )  # aka dropout_rates_mean
            self.q0_log_var = tf.Variable(glorot([n_filters]) * std_init + mean_init)
            self.rsr_M = tf.Variable(glorot([n_filters]))  # var_r_aux
            self.apvar_M = tf.Variable(glorot([n_filters]))  # apvar_r_aux
            self.rsri_M = tf.Variable(glorot([n_filters]))  # var_r_auxi

        self.prior_var_r_p = tf.Variable(
            glorot([self.input_dim]) * std_init + np.log(self.prior_var_w),
            trainable=self.learn_p,
        )
        self.prior_var_r_p_bias = tf.Variable(
            glorot([1]) * std_init + np.log(self.prior_var_b), trainable=self.learn_p,
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
            return tf.ones([batch_size, self.n_filters]), log_dets

        q0_mean = tf.stack([self.q0_mean] * batch_size)
        epsilon = tf.random.normal([batch_size, self.n_filters])
        q0_var = tf.exp(self.q0_log_var)
        z_samples = q0_mean + tf.sqrt(q0_var) * epsilon

        if self.n_flows_q > 0:
            z_samples, log_dets = self.flow_q.forward(z_samples)

        return z_samples, log_dets

    def get_mean_var(self, x):
        std_w = tf.clip_by_value(tf.exp(self.log_std_W), 0, self.max_std)
        var_w = tf.square(std_w)
        var_b = tf.clip_by_value(tf.exp(self.log_var_bias), 0, self.max_std ** 2)

        conv_args = {"strides": self.strides, "padding": self.padding}

        # Perform cross-correlation.
        mean_W_out = tf.nn.conv2d(input=x, filters=self.mean_W, **conv_args)
        var_w_out = tf.nn.conv2d(input=tf.square(x), filters=var_w, **conv_args)
        return mean_W_out + self.mean_b, var_w_out + var_b

    def kl_div(self):
        z_sample, log_dets = self.sample_z(1)

        std_w = tf.exp(self.log_std_W)
        std_w = tf.reshape(std_w, [-1, self.n_filters])
        mu_w = tf.reshape(self.mean_W, [-1, self.n_filters])
        Mtilde = mu_w * z_sample
        mean_b = self.mean_b * z_sample
        Vtilde = tf.square(std_w)
        # outer product
        iUp = tf.stack([tf.exp(self.prior_var_r_p)] * self.n_filters, axis=1)

        log_q = 0
        if self.use_z:
            log_q = -0.5 * tf.reduce_sum(tf.math.log(2 * np.pi) + self.q0_log_var + 1)
            log_q -= log_dets[0]

        kldiv_w = 0.5 * tf.reduce_sum(
            tf.math.log(iUp)
            - tf.math.log(std_w)
            + (Vtilde + tf.square(Mtilde)) / iUp
            - 1
        )
        kldiv_b = 0.5 * tf.reduce_sum(
            self.prior_var_r_p_bias
            - self.log_var_bias
            + (tf.exp(self.log_var_bias) + tf.square(mean_b))
            / tf.exp(self.prior_var_r_p_bias)
            - 1
        )

        log_r = 0
        if self.use_z:
            apvar_M = self.apvar_M
            mean_w = tf.linalg.matvec(Mtilde, apvar_M)
            var_w = tf.linalg.matvec(Vtilde, tf.square(apvar_M))
            epsilon = tf.random.normal([self.input_dim])
            a = mean_w + tf.sqrt(var_w) * epsilon
            mu_b = tf.reduce_sum(mean_b * apvar_M)
            var_b = tf.reduce_sum(tf.exp(self.log_var_bias) * tf.square(apvar_M))
            a += mu_b + tf.sqrt(var_b) * tf.random.normal([])

            w__ = tf.reduce_mean(tf.tensordot(a, self.rsr_M, axes=0), axis=0)
            wv__ = tf.reduce_mean(tf.tensordot(a, self.rsri_M, axes=0), axis=0)

            if self.n_flows_r > 0:
                z_sample, log_r = self.flow_r.forward(z_sample)
                log_r = log_r[0]

            log_r += 0.5 * tf.reduce_sum(
                -tf.exp(wv__) * tf.square(z_sample - w__)
                - tf.math.log(2 * np.pi)
                + wv__
            )

        return kldiv_w + kldiv_b - log_r + log_q

    def call(self, x):
        z_samples, _ = self.sample_z(x.shape[0])
        mean, var = self.get_mean_var(x)

        mu_out = mean * z_samples[:, None, None, :]  # insert singleton dims
        epsilon = tf.random.normal(mu_out.shape)
        sigma_out = tf.sqrt(var) * epsilon

        return mu_out + sigma_out
