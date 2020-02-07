import numpy as np
import tensorflow as tf

from flows import RNVPFlow
from utils import rand_mat


class Conv2DNF(tf.keras.layers.Layer):
    """2D convolutional layer with a normalizing flow approximate
    posterior over the weights. Prior is a standard normal.
    """

    def __init__(
        self,
        nb_filter,
        nb_row,
        nb_col,
        activation=tf.identity,
        border_mode="SAME",
        subsample=(1, 1, 1, 1),
        n_flows_q=2,
        n_flows_r=2,
        learn_p=False,
        use_z=True,
        prior_var_w=1,
        prior_var_b=1,
        flow_dim_h=50,
        thres_std=1,
        var_scale=1,
        **kwargs,
    ):
        self.nb_filter = nb_filter
        self.nb_row = nb_row
        self.nb_col = nb_col
        self.border_mode = border_mode
        self.subsample = subsample
        self.thres_std = thres_std
        self.var_scale = var_scale
        self.flow_dim_h = flow_dim_h
        self.learn_p = learn_p
        self.prior_var_w = prior_var_w
        self.prior_var_b = prior_var_b
        self.n_flows_q = n_flows_q
        self.n_flows_r = n_flows_r
        self.use_z = use_z
        self.activation = activation
        super().__init__(**kwargs)

    def build(self, input_shape):
        stack_size = input_shape[-1]
        vscale = self.var_scale
        self.W_shape = (self.nb_row, self.nb_col, stack_size, self.nb_filter)
        self.input_dim = self.nb_col * stack_size * self.nb_row

        self.mu_w = rand_mat(self.W_shape, name="mean_W")
        self.log_std_W = rand_mat(
            self.W_shape, mu=-9, name="log_std_W", var_scale=vscale
        )
        self.mu_b = tf.Variable(tf.zeros(self.nb_filter), name="mean_bias")
        self.log_var_bias = rand_mat(
            [self.nb_filter], mu=-9, name="log_var_bias", var_scale=vscale
        )

        if self.use_z:
            self.qzero_mean = rand_mat(
                [self.nb_filter],
                name="dropout_rates_mean",
                mu=1 if self.n_flows_q == 0 else 0,
            )
            self.qzero_log_var = rand_mat(
                [self.nb_filter], name="q0_log_var", mu=np.log(0.1), var_scale=vscale,
            )
            self.rsr_M = rand_mat([self.nb_filter], name="var_r_aux")
            self.apvar_M = rand_mat([self.nb_filter], name="apvar_r_aux")
            self.rsri_M = rand_mat([self.nb_filter], name="var_r_auxi")

        self.pvar = rand_mat(
            [self.input_dim],
            mu=np.log(self.prior_var_w),
            name="prior_var_r_p",
            var_scale=vscale,
            trainable=self.learn_p,
        )
        self.pvar_bias = rand_mat(
            [1],
            mu=np.log(self.prior_var_b),
            name="prior_var_r_p_bias",
            var_scale=vscale,
            trainable=self.learn_p,
        )

        if self.n_flows_r > 0:
            self.flow_r = RNVPFlow(
                self.nb_filter,
                n_flows=self.n_flows_r,
                name=self.name + "_flow_r",
                dim_h=2 * self.flow_dim_h,
            )

        if self.n_flows_q > 0:
            self.flow_q = RNVPFlow(
                self.nb_filter,
                n_flows=self.n_flows_q,
                name=self.name + "_flow_q",
                dim_h=self.flow_dim_h,
            )

    def sample_z(self, size_M=1, sample=True):
        logdets = tf.zeros(size_M)
        if not self.use_z:
            return tf.ones([size_M, self.nb_filter]), logdets
        qm0 = tf.exp(self.qzero_log_var)
        isample_M = tf.tile(tf.expand_dims(self.qzero_mean, 0), [size_M, 1])
        eps = tf.random.normal(tf.stack([size_M, self.nb_filter]))
        sample_M = isample_M + tf.sqrt(qm0) * eps if sample else isample_M

        if self.n_flows_q > 0:
            sample_M, logdets = self.flow_q.forward(sample_M, sample=sample)

        return sample_M, logdets

    def get_mean_var(self, x):
        std_w = tf.clip_by_value(tf.exp(self.log_std_W), 0, self.thres_std)
        var_w = tf.square(std_w)
        var_b = tf.clip_by_value(tf.exp(self.log_var_bias), 0, self.thres_std ** 2)

        # formally we do cross-correlation here
        mu_w_out = tf.nn.conv2d(
            input=x,
            filters=self.mu_w,
            strides=self.subsample,
            padding=self.border_mode,
        )
        var_w_out = tf.nn.conv2d(
            input=tf.square(x),
            filters=var_w,
            strides=self.subsample,
            padding=self.border_mode,
        )
        return mu_w_out + self.mu_b, var_w_out + var_b

    def kl_div(self):
        M, logdets = self.sample_z()
        logdets = logdets[0]
        M = tf.squeeze(M)

        std_w = tf.exp(self.log_std_W)
        std_w = tf.reshape(std_w, [-1, self.nb_filter])
        mu = tf.reshape(self.mu_w, [-1, self.nb_filter])
        Mtilde = mu * tf.expand_dims(M, 0)
        mbias = self.mu_b * M
        Vtilde = tf.square(std_w)
        # outer product
        iUp = tf.tensordot(tf.exp(self.pvar), tf.ones(self.nb_filter), axes=0)

        logqm = 0
        if self.use_z:
            logqm = -tf.reduce_sum(
                0.5 * (tf.math.log(2 * np.pi) + self.qzero_log_var + 1)
            )
            logqm -= logdets

        kldiv_w = tf.reduce_sum(
            0.5 * tf.math.log(iUp)
            - tf.math.log(std_w)
            + ((Vtilde + tf.square(Mtilde)) / (2 * iUp))
            - 0.5
        )
        kldiv_bias = tf.reduce_sum(
            0.5 * self.pvar_bias
            - 0.5 * self.log_var_bias
            + (tf.exp(self.log_var_bias) + tf.square(mbias))
            / (2 * tf.exp(self.pvar_bias))
            - 0.5
        )

        logrm = 0
        if self.use_z:
            apvar_M = self.apvar_M
            mw = tf.matmul(Mtilde, tf.expand_dims(apvar_M, 1))
            vw = tf.matmul(Vtilde, tf.expand_dims(tf.square(apvar_M), 1))
            eps = tf.expand_dims(tf.random.normal([self.input_dim]), 1)
            a = mw + tf.sqrt(vw) * eps
            mb = tf.reduce_sum(mbias * apvar_M)
            vb = tf.reduce_sum(tf.exp(self.log_var_bias) * tf.square(apvar_M))
            a += mb + tf.sqrt(vb) * tf.random.normal(())

            w__ = tf.reduce_mean(
                tf.tensordot(tf.squeeze(a), self.rsr_M, axes=0), axis=0
            )
            wv__ = tf.reduce_mean(
                tf.tensordot(tf.squeeze(a), self.rsri_M, axes=0), axis=0
            )

            if self.flow_r is not None:
                M, logrm = self.flow_r.forward(tf.expand_dims(M, 0))
                M = tf.squeeze(M)
                logrm = logrm[0]

            logrm += tf.reduce_sum(
                -0.5 * tf.exp(wv__) * tf.square(M - w__)
                - 0.5 * tf.math.log(2 * np.pi)
                + 0.5 * wv__
            )

        return -kldiv_w + logrm - logqm - kldiv_bias

    def call(self, x, sample=True, **kwargs):
        sample_M, _ = self.sample_z(size_M=x.shape[0], sample=sample)
        sample_M = tf.expand_dims(tf.expand_dims(sample_M, 1), 2)
        mean_out, var_out = self.get_mean_var(x)
        mean_gout = mean_out * sample_M
        if not sample:
            return self.activation(mean_gout)
        var_gout = tf.sqrt(var_out) * tf.random.normal(mean_gout.shape)
        return self.activation(mean_gout + var_gout)
