import numpy as np
import tensorflow as tf


def rand_mat(shape, mean=0, stddev=1, std_init="glorot", **kwargs):
    if len(shape) == 1:
        n_in, n_out = shape[0], 0
    elif len(shape) == 2:
        n_in, n_out = shape
    else:
        n_in, n_out = np.prod(shape[1:]), shape[0]

    # Xavier Glorot initialization (http://proceedings.mlr.press/v9/glorot10a)
    if std_init == "glorot":
        scale = np.sqrt(2 / (n_in + n_out))
    # Kaiming He initialization (https://arxiv.org/abs/1502.01852)
    elif std_init == "he":
        scale = np.sqrt(2 / n_in)
    elif std_init == "const":
        scale = 0.01
    else:
        raise ValueError(f"Unknown value {std_init} for std_init")

    # actual weight initialization
    init_val = tf.random.normal(shape, mean=mean, stddev=stddev * scale)
    return tf.Variable(init_val, **kwargs)
