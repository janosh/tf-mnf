import tensorflow as tf

from layers import Conv2DNF, DenseNF


class NFLeNet(tf.keras.Model):
    def __init__(
        self,
        n_flows_q=2,
        n_flows_r=2,
        use_z=True,
        activation=tf.nn.relu,
        n_classes=10,
        learn_p=False,
        layer_dims=(20, 50, 500),
        flow_dim_h=50,
        thres_std=1,
        prior_var_w=1,
        prior_var_b=1,
        var_scale=1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.activation = activation
        self.n_flows_q = n_flows_q
        self.n_flows_r = n_flows_r
        self.use_z = use_z

        self.flow_dim_h = flow_dim_h
        self.thres_std = thres_std
        self.var_scale = var_scale
        self.learn_p = learn_p
        self.prior_var_w = prior_var_w
        self.prior_var_b = prior_var_b

        layer_config = self.get_layer_config()
        self.conv_1 = Conv2DNF(layer_dims[0], 5, 5, border_mode="VALID", **layer_config)
        self.conv_2 = Conv2DNF(layer_dims[1], 5, 5, border_mode="VALID", **layer_config)
        self.dense_1 = DenseNF(layer_dims[2], **layer_config)
        layer_config["activation"] = tf.nn.softmax
        self.dense_2 = DenseNF(n_classes, **layer_config)

    def get_layer_config(self):
        keys = [
            "n_flows_q",
            "n_flows_r",
            "use_z",
            "learn_p",
            "prior_var_w",
            "prior_var_b",
            "flow_dim_h",
            "thres_std",
            "activation",
            "var_scale",
        ]
        return {key: getattr(self, key) for key in keys}

    def call(self, X, sample=True):
        X = self.conv_1(X, sample=sample)
        X = tf.nn.max_pool2d(
            X, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME"
        )
        X = self.conv_2(X, sample=sample)
        X = tf.nn.max_pool2d(
            X, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME"
        )
        X = tf.keras.layers.Flatten()(X)
        X = self.dense_1(X, sample=sample)
        return self.dense_2(X, sample=sample)

    def kl_div(self):
        """Compute current KL divergence of the whole model.
        Can be used as a regularization term during training.
        """
        return sum([l.kl_div() for l in self.layers])
