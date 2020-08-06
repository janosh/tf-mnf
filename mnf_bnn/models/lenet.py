import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPool2D


class LeNet(tf.keras.Sequential):
    """Just your regular LeNet."""

    def __init__(self, **kwargs):
        c1 = Conv2D(
            filters=32,
            kernel_size=(5, 5),
            padding="same",
            activation="relu",
            input_shape=(28, 28, 1),
        )
        mp1 = MaxPool2D(strides=2)
        c2 = Conv2D(filters=48, kernel_size=(5, 5), activation="relu")
        mp2 = MaxPool2D(strides=2)
        f = Flatten()
        d1 = Dense(256, activation="relu")
        d2 = Dense(84, activation="relu")
        d3 = Dense(10, activation="softmax")
        super().__init__([c1, mp1, c2, mp2, f, d1, d2, d3], **kwargs)
