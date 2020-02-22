# %%
import argparse
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from scipy import ndimage
from tqdm import tqdm

from models import LeNet, NFLeNet

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
parser.add_argument("-learning_rate", type=float, default=0.001)
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

# Create 10000-sample validation set. Leaves 50000 samples for training.
(X_train, X_val), (y_train, y_val) = [np.split(x, [50000]) for x in [X_train, y_train]]

tf.random.set_seed(flags.seed)
np.random.seed(flags.seed)

# %%
nf_lenet = NFLeNet(
    n_flows_q=flags.n_flows_q,
    n_flows_r=flags.n_flows_r,
    use_z=flags.use_z,
    learn_p=flags.learn_p,
    max_std=flags.max_std,
    flow_h_sizes=flags.flow_h_sizes,
    std_init=flags.std_init,
)

optimizer = tf.optimizers.Adam(flags.learning_rate)

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
    kl_loss = nf_lenet.kl_div() / (2 * flags.batch_size)

    tf.summary.scalar("negative log-likelihood", entropic_loss)
    tf.summary.scalar("KL regularization loss", kl_loss)

    return entropic_loss + kl_loss


@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        # We could draw multiple posterior samples here to get unbiased Monte Carlo
        # estimate for the NLL which would decrease training variance but slow us down.
        preds = nf_lenet(images)
        loss = loss_fn(labels, preds)
        tf.summary.scalar("VI lower bound loss (NLL + KL)", loss)
    grads = tape.gradient(loss, nf_lenet.trainable_variables)
    optimizer.apply_gradients(zip(grads, nf_lenet.trainable_variables))

    train_acc = tf.reduce_mean(tf.metrics.categorical_accuracy(labels, preds))
    return train_acc


def train_nf_lenet():
    for epoch in range(flags.epochs):
        for j in tqdm(
            range(flags.steps_per_epoch), desc=f"epoch {epoch + 1}/{flags.epochs}"
        ):
            batch = np.random.choice(len(X_train), flags.batch_size, replace=False)
            tf.summary.experimental.set_step(optimizer.iterations)
            train_acc = train_step(X_train[batch], y_train[batch])
            tf.summary.scalar("training accuracy", train_acc)

        # Accuracy estimated by single call for speed. Would be more accurate to
        # approximately integrate over the parameter posteriors by averaging across
        # multiple calls.
        y_val_pred = nf_lenet(X_val)
        val_acc = tf.reduce_mean(tf.metrics.categorical_accuracy(y_val, y_val_pred))

        tf.summary.scalar("validation accuracy", val_acc)
        print(f"Validation accuracy: {val_acc:.4g}")


# %%
@tf.function
def predict_nf_lenet(X=X_test, n_samples=flags.test_samples):
    preds = []
    for i in tqdm(range(n_samples), desc="Sampling"):
        preds.append(nf_lenet(X))
    return tf.squeeze(preds)


# %%
log_writer = tf.summary.create_file_writer(
    flags.logdir + datetime.now().strftime("%Y.%m.%d-%H:%M:%S")
)
log_writer.set_as_default()

train_nf_lenet()

# %%
# Remove image's channel dimension.
pic4 = X_test[4][..., 0]
pic9 = X_test[12][..., 0]
nrows = ncols = 3

# %%
fig1 = plt.figure(figsize=(12, 8))
fig2 = plt.figure(figsize=(16, 12))
for i in range(1, 10):
    pic9_rot = ndimage.rotate(pic9, i * 20, reshape=False)
    ax1 = fig1.add_subplot(nrows, ncols, i)
    ax1.imshow(pic9_rot, cmap="gray")

    # Insert batch and channel dimension.
    y_pred = predict_nf_lenet(pic9_rot[None, ..., None])
    df = pd.DataFrame(y_pred.numpy()).melt(var_name="digit", value_name="softmax")
    fig2.add_subplot(nrows, ncols, i, ylim=[None, 1])
    # scale="count": Width of violins given by the number of observations in that bin.
    # cut=0: Limit the violin range to the range of observed data.
    sns.violinplot(data=df, x="digit", y="softmax", scale="count", cut=0)


# %%
lenet = LeNet()
lenet.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
lenet_hist = lenet.fit(X_train, y_train)

# %%
fig1 = plt.figure(figsize=(12, 8))
fig2 = plt.figure(figsize=(12, 8))
for i in range(1, 10):
    pic9_rot = ndimage.rotate(pic9, i * 20, reshape=False)
    ax1 = fig1.add_subplot(nrows, ncols, i)
    ax1.imshow(pic9_rot, cmap="gray")

    [y_pred] = lenet.predict(pic9_rot[None, ..., None])
    ax2 = fig2.add_subplot(nrows, ncols, i, ylim=[None, 1.1], xticks=range(10))
    ax2.bar(range(10), y_pred)
