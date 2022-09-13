import numpy as np
import tensorflow as tf

from tf_mnf.flows import IAF, NormalizingFlow


class MNFConv2D(tf.keras.layers.Layer):
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
        super().__init__(**kwargs)
        self.n_filters = n_filters
        self.kernel_size = (
            [kernel_size, kernel_size] if type(kernel_size) == int else kernel_size
        )
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
        self.log_var_b = tf.Variable(glorot([n_filters]) * std_init + mean_init)

        if self.use_z:
            # q0_mean has similar function to a dropout rate as it determines the
            # mean of the multiplicative noise z_k in eq. 5.
            self.q0_mean = tf.Variable(
                glorot([n_filters]) + (0 if self.n_flows_q > 0 else 1)
            )
            self.q0_log_var = tf.Variable(glorot([n_filters]) * std_init + mean_init)

            self.r0_mean = tf.Variable(glorot([n_filters]))
            self.r0_log_var = tf.Variable(glorot([n_filters]))
            self.r0_apvar = tf.Variable(glorot([n_filters]))

        self.prior_var_r_p = tf.Variable(
            glorot([self.input_dim]) * std_init + np.log(self.prior_var_w),
            trainable=self.learn_p,
        )
        self.prior_var_r_p_bias = tf.Variable(
            glorot([1]) * std_init + np.log(self.prior_var_b),
            trainable=self.learn_p,
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

        q0_mean = tf.tile(self.q0_mean[None, :], [batch_size, 1])
        epsilon = tf.random.normal([batch_size, self.n_filters])
        q0_var = tf.exp(self.q0_log_var)
        z_samples = q0_mean + tf.sqrt(q0_var) * epsilon

        if self.n_flows_q > 0:
            z_samples, log_dets = self.flow_q.forward(z_samples)
            z_samples = z_samples[-1]

        return z_samples, log_dets

    def kl_div(self):
        z_sample, log_det_q = self.sample_z(1)

        std_w = tf.exp(self.log_std_W)
        std_w = tf.reshape(std_w, [-1, self.n_filters])
        mu_w = tf.reshape(self.mean_W, [-1, self.n_filters])
        Mtilde = mu_w * z_sample
        mean_b = self.mean_b * z_sample
        Vtilde = tf.square(std_w)
        # Stacking yields same result as outer product with ones. See eqs. 11, 12.
        iUp = tf.stack([tf.exp(self.prior_var_r_p)] * self.n_filters, axis=1)

        kl_div_w = 0.5 * tf.reduce_sum(
            tf.math.log(iUp)
            - tf.math.log(std_w)
            + (Vtilde + tf.square(Mtilde)) / iUp
            - 1
        )
        kl_div_b = 0.5 * tf.reduce_sum(
            self.prior_var_r_p_bias
            - self.log_var_b
            + (tf.exp(self.log_var_b) + tf.square(mean_b))
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

            mean_w = tf.linalg.matvec(Mtilde, self.r0_apvar)
            var_w = tf.linalg.matvec(Vtilde, tf.square(self.r0_apvar))
            epsilon = tf.random.normal([self.input_dim])
            # For convolutional layers, linear mappings empirically work better than
            # tanh non-linearity. Hence the removal of a = tf.tanh(a). Christos Louizos
            # confirmed this in https://github.com/AMLab-Amsterdam/MNF_VBNN/issues/4
            # even though the paper states the use of tanh in conv layers.
            a = mean_w + tf.sqrt(var_w) * epsilon
            # a = tf.tanh(a)
            mu_b = tf.reduce_sum(mean_b * self.r0_apvar)
            var_b = tf.reduce_sum(tf.exp(self.log_var_b) * tf.square(self.r0_apvar))
            a += mu_b + tf.sqrt(var_b) * tf.random.normal([])
            # a = tf.tanh(a)

            # Mean and log variance of the auxiliary normal dist. r(z_T_b|W) in eq. 8.
            mean_r = tf.reduce_mean(tf.tensordot(a, self.r0_mean, axes=0), axis=0)
            log_var_r = tf.reduce_mean(tf.tensordot(a, self.r0_log_var, axes=0), axis=0)

            # Log likelihood of a zero-covariance normal dist: ln N(x | mu, sigma) =
            # -1/2 sum_dims(ln(2 pi) + ln(sigma^2) + (x - mu)^2 / sigma^2)
            log_r += 0.5 * tf.reduce_sum(
                -tf.exp(log_var_r) * tf.square(z_sample - mean_r)
                - tf.math.log(2 * np.pi)
                + log_var_r
            )

        return kl_div_w + kl_div_b - log_r + log_q

    def call(self, x):
        z_samples, _ = self.sample_z(tf.shape(x)[0])

        std_w = tf.clip_by_value(tf.exp(self.log_std_W), 0, self.max_std)
        var_w = tf.square(std_w)
        var_b = tf.clip_by_value(tf.exp(self.log_var_b), 0, self.max_std**2)

        # Perform cross-correlation.
        mean = tf.nn.conv2d(input=x, filters=self.mean_W) + self.mean_b
        var = tf.nn.conv2d(input=tf.square(x), filters=var_w) + var_b

        mu_out = mean * z_samples[:, None, None, :]  # insert singleton dims
        epsilon = tf.random.normal(tf.shape(mu_out))
        sigma_out = tf.sqrt(var) * epsilon

        return mu_out + sigma_out
