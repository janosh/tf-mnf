# %%
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from tf_mnf import models  # ,ROOT
from tf_mnf.evaluate import rot_img


# %%
plt.rcParams["figure.figsize"] = [12, 8]

epochs = 3
batch_size = 64


# %%
# Load MNIST handwritten digits. 60000 images for training, 10000 for testing.
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

# Constrain pixel values to the unit interval [0, 1].
X_train, X_test = (X.astype("float32") / 255 for X in [X_train, X_test])

# Add a channels (aka stacks) dimension.
X_train, X_test = X_train[..., None], X_test[..., None]

# One-hot encode the labels.
y_train, y_test = (tf.keras.utils.to_categorical(y, 10) for y in [y_train, y_test])

# ensure reproducible results
tf.random.set_seed(0)
np.random.seed(0)


# %%
mnf_lenet = models.MNFLeNet(
    max_std=1,  # Max stddev for layer weights. Larger values are clipped at call time
    flow_h_sizes=[50],  # How many and what size of dense layers to use for the NF nets
    std_init=1e1,  # Scaling factor for initial stddev of tf.Variables
)

adam = tf.optimizers.Adam(1e-3)


# %%
# We minimize the negative log-likelihood, i.e. maximize the log-likelihood of
# observed data (X_train, y_train) under the model
def loss_fn(labels, preds):
    # entropic_loss = multiclass cross entropy = negative log-likelihood.
    cross_entropy = tf.losses.categorical_crossentropy(labels, preds)
    entropic_loss = tf.reduce_mean(cross_entropy)
    # The weighting factor dividing the KL divergence can be used as a hyperparameter.
    # Decreasing it makes learning more difficult but prevents model overfitting. If
    # not seen as hyperparameter, it should be applied once per epoch, i.e. divided by
    # the number of samples in one epoch.
    kl_loss = mnf_lenet.kl_div() / len(X_train)
    loss = entropic_loss + kl_loss

    tf.summary.scalar("negative log-likelihood", entropic_loss)
    tf.summary.scalar("KL divergence", kl_loss)
    tf.summary.scalar("loss", loss)

    return loss


# %%
mnf_lenet.compile(loss=loss_fn, optimizer=adam, metrics=["accuracy"])


# %%
nf_hist = mnf_lenet.fit(
    X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1
)


# %%
img9 = X_test[12]
test_samples = 500
rot_img(lambda x: mnf_lenet(x.repeat(test_samples, axis=0)).numpy(), img9, axes=[1, 0])
# plt.savefig(ROOT + "/assets/rot-9-mnf-lenet.pdf")


# %%
lenet = models.LeNet()
lenet.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])


# %%
lenet_hist = lenet.fit(X_train, y_train, epochs=epochs)


# %%
rot_img(lambda x: lenet(x).numpy(), img9, plot_type="bar", axes=[1, 0])
# plt.savefig(ROOT + "/assets/rot-9-lenet.pdf")

# Below is code for low-level training with tf.GradientTape(). Slower and more verbose
# but easier to debug, especially with @tf.function commented out.


# %%
@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        # We could draw multiple posterior samples here to get unbiased Monte Carlo
        # estimate for the NLL which would decrease training variance but slow us down.
        preds = mnf_lenet(images)
        loss = loss_fn(labels, preds)
        tf.summary.scalar("VI lower bound loss (NLL + KL)", loss)
    grads = tape.gradient(loss, mnf_lenet.trainable_variables)
    adam.apply_gradients(zip(grads, mnf_lenet.trainable_variables))

    train_acc = tf.reduce_mean(tf.metrics.categorical_accuracy(labels, preds))
    return loss, train_acc


def train_mnf_lenet(log_every=50):
    for epoch in range(epochs):
        idx = np.arange(len(y_train))
        np.random.shuffle(idx)
        batches = np.split(idx, len(y_train) / batch_size)
        pbar = tqdm(batches, desc=f"epoch {epoch + 1}/{epochs}")
        for step, batch in enumerate(pbar):
            loss, train_acc = train_step(X_train[batch], y_train[batch])

            if step % log_every == 0:
                tf.summary.experimental.set_step(adam.iterations)
                tf.summary.scalar("accuracy/training", train_acc)
                pbar.set_postfix(loss=f"{loss:.4}", train_acc=f"{train_acc:.4}")


log_writer = tf.summary.create_file_writer(
    f"logs/lenet/{datetime.now():%m.%d-%H:%M:%S}"
)
log_writer.set_as_default()

train_mnf_lenet()
