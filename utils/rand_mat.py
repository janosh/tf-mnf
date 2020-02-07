import numpy as np
import tensorflow as tf


def rand_mat(shape, mu=0, var_init="he2", var_scale=1, **kwargs):
    if len(shape) == 1:
        dim_in, dim_out = shape[0], 0
    elif len(shape) == 2:
        dim_in, dim_out = shape
    else:
        dim_in, dim_out = np.prod(shape[1:]), shape[0]

    # Xavier Glorot initialization (http://proceedings.mlr.press/v9/glorot10a)
    if var_init == "glorot":
        bound = np.sqrt(1 / dim_in)
    elif var_init == "glorot2":
        bound = np.sqrt(2 / (dim_in + dim_out))
    # Kaiming He initialization (https://arxiv.org/abs/1502.01852)
    elif var_init == "he":
        bound = np.sqrt(2 / dim_in)
    elif var_init == "he2":
        bound = np.sqrt(4 / (dim_in + dim_out))
    elif var_init == "regular":
        bound = 0.01  # sigma_init
    else:
        raise ValueError("Unknown variance initialization for rand_mat")

    # actual weight initialization
    init_val = tf.random.normal(shape, mean=mu, stddev=var_scale * bound)
    return tf.Variable(init_val, **kwargs)
