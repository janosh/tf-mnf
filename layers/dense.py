import numpy as np
import tensorflow as tf

from flows import RNVPFlow
from utils import rand_mat


class DenseNF(tf.keras.layers.Layer):
    """Fully connected layer with a normalizing flow approximate posterior over the weights.
    Prior is a standard normal.
    """

    def __init__(
        self,
        dim_out,
        activation=tf.identity,
        n_flows_q=2,
        n_flows_r=2,
        learn_p=False,
        use_z=True,
        prior_var_w=1,
        flow_dim_h=50,
        prior_var_b=1,
        thres_std=1,
        var_scale=1,
        **kwargs,
    ):
        self.dim_out = dim_out
        self.learn_p = learn_p
        self.prior_var_w = prior_var_w
        self.prior_var_b = prior_var_b
        self.thres_std = thres_std
        self.var_scale = var_scale
        self.n_flows_q = n_flows_q
        self.n_flows_r = n_flows_r
        self.use_z = use_z
        self.flow_dim_h = flow_dim_h
        self.activation = activation
        super().__init__(**kwargs)

    def build(self, input_shape):
        d_in = self.dim_in = input_shape[-1]
        d_out = self.dim_out
        vscale = self.var_scale

        self.mu_w = rand_mat((d_in, d_out), name="mean_W")
        self.log_std_W = rand_mat(
            (d_in, d_out), mu=-9, name="log_std_W", var_scale=vscale
        )
        self.mu_bias = tf.Variable(tf.zeros(d_out), name="mean_bias")
        self.log_var_bias = rand_mat(
            [d_out], mu=-9, name="log_var_bias", var_scale=vscale
        )

        if self.use_z:
            self.qzero_mean = rand_mat(
                [d_in], name="q0_mean", mu=1 if self.n_flows_q == 0 else 0,
            )
            self.qzero_log_var = rand_mat(
                [d_in], mu=np.log(0.1), name="q0_log_var", var_scale=vscale
            )
            self.rsr_M = rand_mat([d_in], name="var_r_aux")
            self.apvar_M = rand_mat([d_in], name="apvar_r_aux")
            self.rsri_M = rand_mat([d_in], name="var_r_auxi")

        self.pvar = rand_mat(
            [d_in],
            mu=np.log(self.prior_var_w),
            name="prior_var_r_p",
            trainable=self.learn_p,
            var_scale=vscale,
        )
        self.pvar_bias = rand_mat(
            [1],
            mu=np.log(self.prior_var_b),
            name="prior_var_r_p_bias",
            trainable=self.learn_p,
            var_scale=vscale,
        )

        if self.n_flows_r > 0:
            self.flow_r = RNVPFlow(
                d_in,
                n_flows=self.n_flows_r,
                name=self.name + "_flow_r",
                dim_h=2 * self.flow_dim_h,
            )

        if self.n_flows_q > 0:
            self.flow_q = RNVPFlow(
                d_in,
                n_flows=self.n_flows_q,
                name=self.name + "_flow_q",
                dim_h=self.flow_dim_h,
            )

    def sample_z(self, size_M=1, sample=True):
        logdets = tf.zeros(size_M)
        if not self.use_z:
            return tf.ones([size_M, self.dim_in]), logdets

        qm0 = tf.exp(self.qzero_log_var)
        isample_M = tf.tile(tf.expand_dims(self.qzero_mean, 0), [size_M, 1])
        eps = tf.random.normal(tf.stack([size_M, self.dim_in]))
        sample_M = isample_M + tf.sqrt(qm0) * eps if sample else isample_M

        if self.n_flows_q > 0:
            sample_M, logdets = self.flow_q.forward(sample_M, sample=sample)

        return sample_M, logdets

    def kl_div(self):
        M, logdets = self.sample_z()
        logdets = logdets[0]
        M = tf.squeeze(M)

        if len(M.shape) == 0:
            Mexp = M
        else:
            Mexp = tf.expand_dims(M, 1)

        Mtilde = Mexp * self.mu_w
        Vtilde = tf.square(tf.exp(self.log_std_W))
        # outer product
        iUp = tf.tensordot(tf.exp(self.pvar), tf.ones(self.dim_out), axes=0)

        logqm = 0
        if self.use_z:
            # Compute entropy of the initial distribution q(z_0).
            # This is independent of the actual sample z_0.
            logqm = -tf.reduce_sum(
                0.5 * (tf.math.log(2 * np.pi) + self.qzero_log_var + 1)
            )
            logqm -= logdets

        kldiv_w = tf.reduce_sum(
            0.5 * tf.math.log(iUp)
            - self.log_std_W
            + ((Vtilde + tf.square(Mtilde)) / (2 * iUp))
            - 0.5
        )
        kldiv_bias = tf.reduce_sum(
            0.5 * self.pvar_bias
            - 0.5 * self.log_var_bias
            + (tf.exp(self.log_var_bias) + tf.square(self.mu_bias))
            / (2 * tf.exp(self.pvar_bias))
            - 0.5
        )

        if self.use_z:
            apvar_M = self.apvar_M
            # shared network for hidden layer
            mw = tf.matmul(tf.expand_dims(apvar_M, 0), Mtilde)
            eps = tf.expand_dims(tf.random.normal([self.dim_out]), 0)
            varw = tf.matmul(tf.square(tf.expand_dims(apvar_M, 0)), Vtilde)
            a = tf.tanh(mw + tf.sqrt(varw) * eps)
            # split at output layer
            if len(tf.squeeze(a).shape) != 0:
                w__ = tf.tensordot(self.rsr_M, tf.squeeze(a), axes=0)
                w__ = tf.reduce_mean(w__, axis=1)
                wv__ = tf.tensordot(self.rsri_M, tf.squeeze(a), axes=0)
                wv__ = tf.reduce_mean(wv__, axis=1)
            else:
                w__ = self.rsr_M * tf.squeeze(a)
                wv__ = self.rsri_M * tf.squeeze(a)

            logrm = 0
            if self.flow_r is not None:
                M, logrm = self.flow_r.forward(tf.expand_dims(M, 0))
                M = tf.squeeze(M)
                logrm = logrm[0]

            logrm += tf.reduce_sum(
                -0.5 * tf.exp(wv__) * tf.square(M - w__)
                - 0.5 * tf.math.log(2 * np.pi)
                + 0.5 * wv__
            )
        else:
            logrm = 0

        return -kldiv_w - kldiv_bias + logrm - logqm

    def call(self, x, sample=True, **kwargs):
        std_mg = tf.clip_by_value(tf.exp(self.log_std_W), 0, self.thres_std)
        var_mg = tf.square(std_mg)
        sample_M, _ = self.sample_z(size_M=x.shape[0], sample=sample)
        xt = x * sample_M

        mu_out = tf.matmul(xt, self.mu_w) + self.mu_bias
        varin = tf.matmul(tf.square(x), var_mg) + tf.clip_by_value(
            tf.exp(self.log_var_bias), 0, self.thres_std ** 2
        )
        xin = tf.sqrt(varin)
        sigma_out = xin * tf.random.normal(mu_out.shape)

        output = mu_out + sigma_out if sample else mu_out
        return self.activation(output)
