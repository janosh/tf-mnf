# %%
import argparse
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from tf_mnf import models  # ,ROOT
from tf_mnf.evaluate import rot_img

# %%
plt.rcParams["figure.figsize"] = [12, 8]

parser = argparse.ArgumentParser(allow_abbrev=False)
# TensorBoard log directory
parser.add_argument("-logdir", type=str, default="logs/lenet")
parser.add_argument("-epochs", type=int, default=3)
parser.add_argument("-batch_size", type=int, default=64)
# Whether to use auxiliary random variable z ~ q(z) to increase expressivity of
# weight posteriors q(W|z).
parser.add_argument("-use_z", action="store_false")
parser.add_argument("-n_flows_q", type=int, default=2)
parser.add_argument("-n_flows_r", type=int, default=2)
# Random seed to ensure reproducible results.
parser.add_argument("-seed", type=int, default=0)
parser.add_argument("-learning_rate", type=float, default=1e-3)
# Maximum stddev for layer weights. Larger values will be clipped at call time.
parser.add_argument("-max_std", type=float, default=1)
# How many and what size of dense layers to use in the multiplicative normalizing flow.
parser.add_argument("-flow_h_sizes", type=int, default=[50])
# How many predictions to make at test time. More yield better uncertainty estimates.
parser.add_argument("-test_samples", type=int, default=50)
parser.add_argument("-learn_p", action="store_true")
# Scaling factor for initial stddev of Glorot-normal initialized tf.Variables.
parser.add_argument("-std_init", type=float, default=1e1)
flags, _ = parser.parse_known_args()


# %%
# Load MNIST handwritten digits. 60000 images for training, 10000 for testing.
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

# Constrain pixel values to the unit interval [0, 1].
X_train, X_test = [X.astype("float32") / 255 for X in [X_train, X_test]]

# Add a channels (aka stacks) dimension.
X_train, X_test = X_train[..., None], X_test[..., None]

# One-hot encode the labels.
y_train, y_test = [tf.keras.utils.to_categorical(y, 10) for y in [y_train, y_test]]

tf.random.set_seed(flags.seed)
np.random.seed(flags.seed)


# %%
layer_args = [
    *["use_z", "n_flows_q", "n_flows_r", "learn_p"],
    *["max_std", "flow_h_sizes", "std_init"],
]
layer_args = {key: getattr(flags, key) for key in layer_args}
mnf_lenet = models.MNFLeNet(**layer_args)

adam = tf.optimizers.Adam(flags.learning_rate)


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
    X_train,
    y_train,
    epochs=flags.epochs,
    batch_size=flags.batch_size,
    validation_split=0.1,
)


# %%
img9 = X_test[12]
rot_img(lambda x: mnf_lenet(x.repeat(500, axis=0)).numpy(), img9, axes=[1, 0])
# plt.savefig(ROOT + "/assets/rot-9-mnf-lenet.pdf")


# %%
lenet = models.LeNet()
lenet.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])


# %%
lenet_hist = lenet.fit(X_train, y_train, epochs=flags.epochs)


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
    for epoch in range(flags.epochs):
        idx = np.arange(len(y_train))
        np.random.shuffle(idx)
        batches = np.split(idx, len(y_train) / flags.batch_size)
        pbar = tqdm(batches, desc=f"epoch {epoch + 1}/{flags.epochs}")
        for step, batch in enumerate(pbar):
            loss, train_acc = train_step(X_train[batch], y_train[batch])

            if step % log_every == 0:
                tf.summary.experimental.set_step(adam.iterations)
                tf.summary.scalar("accuracy/training", train_acc)
                pbar.set_postfix(loss=f"{loss:.4}", train_acc=f"{train_acc:.4}")


log_writer = tf.summary.create_file_writer(
    f"{flags.logdir}/{datetime.now():%m.%d-%H:%M:%S}"
)
log_writer.set_as_default()

train_mnf_lenet()
