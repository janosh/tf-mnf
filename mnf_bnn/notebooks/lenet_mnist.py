# %%
import argparse
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from scipy.ndimage import rotate
from tqdm import tqdm

from mnf_bnn.models import LeNet, MNFLeNet

# %%
parser = argparse.ArgumentParser(allow_abbrev=False)
# TensorBoard log directory
parser.add_argument("-logdir", type=str, default="logs/lenet/")
parser.add_argument("-epochs", type=int, default=3)
parser.add_argument("-steps_per_epoch", type=int, default=300)
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
mnf_lenet = MNFLeNet(**layer_args)

adam = tf.optimizers.Adam(flags.learning_rate)


# %%
def loss_fn(labels, preds):
    # Returns minus the evidence lower bound (ELBO) which we minimize. This implicitly
    # maximizes the log-likelihood of observed data (X_train, y_train) under the model.

    # entropic_loss is the multiclass cross entropy aka negative log-likelihood.
    cross_entropy = tf.losses.categorical_crossentropy(labels, preds)
    entropic_loss = tf.reduce_mean(cross_entropy)
    # The weighting factor dividing the KL divergence can be used as a hyperparameter.
    # Decreasing it makes learning more difficult but prevents model overconfidence. If
    # not seen as hyperparameter, it should be applied once per epoch, i.e. divided by
    # the total number of samples in an epoch (batch_size * steps_per_epoch)
    kl_loss = mnf_lenet.kl_div() / (2 * flags.batch_size)

    tf.summary.scalar("negative log-likelihood", entropic_loss)
    tf.summary.scalar("KL regularization loss", kl_loss)

    return entropic_loss + kl_loss


# %%
mnf_lenet.compile(loss=loss_fn, optimizer=adam, metrics=["accuracy"])

fit_args = {k: getattr(flags, k) for k in ["batch_size", "epochs", "steps_per_epoch"]}
nf_hist = mnf_lenet.fit(X_train, y_train, **fit_args, validation_split=0.1)


# %%
pic9 = X_test[12]


# %%
fig, axes = plt.subplots(3, 3, figsize=(12, 8))
for i, ax1 in enumerate(axes.flat):
    pic9_rot = rotate(pic9, i * 20, reshape=False)

    # Insert batch and channel dimension.
    y_pred = mnf_lenet(tf.tile(pic9_rot[None, ...], [50, 1, 1, 1]))
    df = pd.DataFrame(y_pred.numpy()).melt(var_name="digit", value_name="softmax")
    # scale="count": Width of violins given by the number of observations in that bin.
    # cut=0: Limit the violin range to the range of observed data.
    sns.violinplot(data=df, x="digit", y="softmax", scale="count", cut=0, ax=ax1)
    ax1.set(ylim=[None, 1.1])
    ax2 = ax1.inset_axes([0, 0.5, 0.4, 0.4])
    ax2.axis("off")
    ax2.imshow(pic9_rot.squeeze(), cmap="gray")


# %%
lenet = LeNet()
lenet.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
lenet_hist = lenet.fit(X_train, y_train)


# %%
fig, axes = plt.subplots(3, 3, figsize=(12, 8))
for i, ax1 in enumerate(axes.flat):
    pic9_rot = rotate(pic9, i * 20, reshape=False)

    [y_pred] = lenet(pic9_rot[None, ...])
    ax1.bar(range(10), y_pred)
    ax1.set(ylim=[None, 1.1], xticks=range(10))
    ax2 = ax1.inset_axes([0, 0.5, 0.4, 0.4])
    ax2.axis("off")
    ax2.imshow(pic9_rot.squeeze(), cmap="gray")


# Below is code for low-level training with tf.GradienTape. More verbose but easier to
# debug, especially with @tf.function commented out.


# %%
# Create 10000-sample validation set. Leaves 50000 samples for training.
try:
    X_val, y_val  # type: ignore
except NameError:
    X_train, X_val = np.split(X_train, [50000])
    y_train, y_val = np.split(y_train, [50000])


# %%
# @tf.function
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
    return train_acc


def train_mnf_lenet():
    for epoch in range(flags.epochs):
        for j in tqdm(
            range(flags.steps_per_epoch), desc=f"epoch {epoch + 1}/{flags.epochs}"
        ):
            batch = np.random.choice(len(X_train), flags.batch_size, replace=False)
            tf.summary.experimental.set_step(adam.iterations)
            train_acc = train_step(X_train[batch], y_train[batch])
            tf.summary.scalar("training accuracy", train_acc)

        # Accuracy estimated by single call for speed. Would be more accurate to
        # approximately integrate over the parameter posteriors by averaging across
        # multiple calls.
        y_val_pred = mnf_lenet(X_val)
        val_acc = tf.reduce_mean(tf.metrics.categorical_accuracy(y_val, y_val_pred))

        tf.summary.scalar("validation accuracy", val_acc)
        print(f"Validation accuracy: {val_acc:.4g}")


log_writer = tf.summary.create_file_writer(
    flags.logdir + datetime.now().strftime("%Y.%m.%d-%H:%M:%S")
)
log_writer.set_as_default()

train_mnf_lenet()
