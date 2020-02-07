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

from models import NFLeNet, build_lenet

# %%
parser = argparse.ArgumentParser(allow_abbrev=False)
# TensorBoard log directory
parser.add_argument("-logdir", type=str, default="logs/lenet/")
parser.add_argument("-epochs", type=int, default=1)
parser.add_argument("-n_flows_q", type=int, default=2)
parser.add_argument("-n_flows_r", type=int, default=2)
parser.add_argument("-use_z", action="store_true")
parser.add_argument("-seed", type=int, default=1)
parser.add_argument("-learning_rate", type=float, default=0.001)
parser.add_argument("-thres_std", type=float, default=1)
parser.add_argument("-flow_dim_h", type=int, default=50)
parser.add_argument("-test_samples", type=int, default=50)
parser.add_argument("-learn_p", action="store_true")
parser.add_argument("-steps_per_epoch", type=int, default=500)
parser.add_argument("-var_scale", type=float, default=20)
flags, _ = parser.parse_known_args()

# %%
# Load MNIST handwritten digits. 60000 images for training, 10000 for testing.
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

# Constrain pixel values to unit interval.
X_train, X_test = [X.astype("float32") / 255 for X in [X_train, X_test]]

# Add a channels dimension.
X_train, X_test = X_train[..., None], X_test[..., None]

# One-hot encode labels.
y_train, y_test = [tf.keras.utils.to_categorical(y, 10) for y in [y_train, y_test]]

# Create validation set.
(X_train, X_val), (y_train, y_val) = [
    np.split(ds, [50000]) for ds in [X_train, y_train]
]

n_samples = X_train.shape[0]  # number of training samples

tf.random.set_seed(flags.seed)
np.random.seed(flags.seed)

# %%
nf_lenet = NFLeNet(
    n_flows_q=flags.n_flows_q,
    n_flows_r=flags.n_flows_r,
    use_z=flags.use_z,
    learn_p=flags.learn_p,
    thres_std=flags.thres_std,
    flow_dim_h=flags.flow_dim_h,
    var_scale=flags.var_scale,
)

optimizer = tf.optimizers.Adam(learning_rate=flags.learning_rate)

# %%
@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        preds = nf_lenet(images)

        cross_entropy = tf.losses.categorical_crossentropy(
            tf.stop_gradient(labels), preds
        )
        entropic_loss = tf.reduce_mean(cross_entropy)
        tf.summary.scalar(
            "negative log-likelihood (aka multiclass cross entropy)", entropic_loss
        )

        kl_loss = -nf_lenet.kl_div() / n_samples
        tf.summary.scalar("KL regularization loss", kl_loss)

        lowerbound = entropic_loss + kl_loss
        tf.summary.scalar("Lower bound loss (KL + NLL)", lowerbound)
    grads = tape.gradient(lowerbound, nf_lenet.trainable_variables)
    optimizer.apply_gradients(zip(grads, nf_lenet.trainable_variables))

    train_acc = tf.reduce_mean(tf.metrics.categorical_accuracy(labels, preds))
    tf.summary.scalar("training accuracy", train_acc)


def train_nf_lenet():
    idx = np.arange(n_samples)
    for epoch in range(flags.epochs):
        np.random.shuffle(idx)
        for j in tqdm(
            range(flags.steps_per_epoch), desc=f"Epoch {epoch + 1}/{flags.epochs}"
        ):
            batch = np.random.choice(idx, 100)
            tf.summary.experimental.set_step(optimizer.iterations)
            train_step(X_train[batch], y_train[batch])

        # The accuracy here is calculated by a crude MAP for fast evaluation.
        # Would be much accurate to properly integrate over the parameters by
        # averaging across multiple samples.
        y_val_pred = nf_lenet(X_val, sample=False)
        val_acc = tf.reduce_mean(tf.metrics.categorical_accuracy(y_val, y_val_pred))

        tf.summary.scalar("validation accuracy", val_acc)
        print(f"Validation accuracy: {val_acc:.4g}")


# %%
@tf.function
def predict_nf_lenet(X=X_test, samples=flags.test_samples):
    preds = []
    for i in tqdm(range(samples), desc="Sampling"):
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
    ax1 = fig1.add_subplot(nrows, ncols, i)
    ax2 = fig2.add_subplot(nrows, ncols, i)

    pic9_rot = ndimage.rotate(pic9, i * 20, reshape=False)
    ax1.imshow(pic9_rot, cmap="gray")

    # Insert batch and channel dimension.
    y_pred = predict_nf_lenet(pic9_rot[None, ..., None])
    df = pd.DataFrame(y_pred.numpy()).melt(var_name="digit", value_name="softmax")
    # scale="count": Width of violins given by the number of observations in that bin.
    # cut=0: Limit the violin range to the range of observed data.
    sns.violinplot(data=df, x="digit", y="softmax", scale="count", cut=0)

# %%
lenet = build_lenet()
lenet.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
lenet_hist = lenet.fit(X_train, y_train)

# %%
fig1 = plt.figure(figsize=(12, 8))
fig2 = plt.figure(figsize=(12, 8))
for i in range(1, 10):
    ax1 = fig1.add_subplot(nrows, ncols, i)
    ax2 = fig2.add_subplot(nrows, ncols, i)

    pic9_rot = ndimage.rotate(pic9, i * 20, reshape=False)
    ax1.imshow(pic9_rot, cmap="gray")

    [y_pred] = lenet.predict(pic9_rot[None, ..., None])
    ax2.bar(range(10), y_pred)
    ax2.set_ylim(None, 1.1)
    ax2.set_xticks(range(10))
