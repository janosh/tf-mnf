import tensorflow as tf

tfl = tf.keras.layers


class LeNet(tf.keras.Sequential):
    """Just your regular LeNet."""

    def __init__(self, **kwargs):
        c1 = tfl.Conv2D(
            filters=32,
            kernel_size=(5, 5),
            padding="same",
            activation="relu",
            input_shape=(28, 28, 1),
        )
        mp1 = tfl.MaxPool2D(strides=2)
        c2 = tfl.Conv2D(filters=48, kernel_size=(5, 5), activation="relu")
        mp2 = tfl.MaxPool2D(strides=2)
        f = tfl.Flatten()
        d1 = tfl.Dense(256, activation="relu")
        d2 = tfl.Dense(84, activation="relu")
        d3 = tfl.Dense(10, activation="softmax")
        super().__init__([c1, mp1, c2, mp2, f, d1, d2, d3], **kwargs)
