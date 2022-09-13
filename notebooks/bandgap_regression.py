# %%
import argparse
from datetime import datetime

import pandas as pd
import seaborn as sns
import tensorflow as tf
from tqdm import tqdm

from tf_mnf import ROOT, models


# %%
parser = argparse.ArgumentParser(allow_abbrev=False)
# TensorBoard log directory
parser.add_argument("-logdir", default=ROOT + "/logs/bandgap/")
parser.add_argument("-epochs", type=int, default=5)
parser.add_argument("-batch_size", type=int, default=64)
parser.add_argument("-n_flows_q", type=int, default=2)
parser.add_argument("-n_flows_r", type=int, default=2)
parser.add_argument("-use_z", action="store_false")
parser.add_argument("-learn_p", action="store_true")
parser.add_argument("-seed", type=int, default=1)
parser.add_argument("-learning_rate", type=float, default=1e-3)
parser.add_argument("-max_std", type=float, default=1)
parser.add_argument("-flow_h_sizes", type=int, default=[50])
parser.add_argument("-test_samples", type=int, default=50)
parser.add_argument("-std_init", type=float, default=1e-2)
flags, _ = parser.parse_known_args()


# %%
features = pd.read_csv(ROOT + "/data/bandgaps.csv")
composition = features.pop("Composition")
features = features.astype("float32")
bandgaps = features.pop("Eg (eV)")

X_train = features.sample(frac=0.8, random_state=flags.seed)
X_test = features.drop(X_train.index)
y_train, y_test = (bandgaps[X.index] for X in [X_train, X_test])

tf.random.set_seed(flags.seed)


# %%
layer_args = ["use_z", "n_flows_q", "n_flows_r", "learn_p"]
layer_args += ["max_std", "flow_h_sizes", "std_init"]

model = models.MNFFeedForward(
    layer_sizes=[100, 50, 10, 1], **{key: getattr(flags, key) for key in layer_args}
)

adam = tf.optimizers.Adam(flags.learning_rate)


# %%
def loss_fn(y_true, y_pred):
    # Assuming Gaussian L2 loss equivalent to maximum likelihood estimation.
    mse = tf.reduce_mean(tf.losses.mse(y_true, y_pred))
    tf.summary.scalar("MSE", mse)

    # The weighting factor dividing the KL divergence can be used as a hyperparameter.
    # Decreasing it makes learning more difficult but prevents model overconfidence. If
    # not seen as hyperparameter, it should be applied once per epoch, i.e. divided by
    # the total number of samples in an epoch (batch_size * steps_per_epoch)
    kl_loss = model.kl_div() / (2 * flags.batch_size)
    tf.summary.scalar("KL regularization loss", kl_loss)

    return mse + kl_loss


# %%
model.compile(loss=loss_fn, optimizer=adam, metrics=["accuracy"])

fit_args = {k: getattr(flags, k) for k in ["batch_size", "epochs"]}
hist = model.fit(X_train.values, y_train.values, **fit_args)


# %%
def predict(X=X_test.values, n_samples=flags.test_samples):
    # using training=False for layers like BatchNormalization or Dropout that behave
    # differently during inference.
    preds = [model(X, training=False) for _ in tqdm(range(n_samples), desc="Sampling")]
    return tf.squeeze(preds)


# %%
preds = predict().numpy()
preds = pd.DataFrame(preds, columns=composition.loc[y_test.index]).T.reset_index()
preds["Eg (eV)"] = y_test.values


# %%
melted = preds.melt(id_vars=["Composition", "Eg (eV)"], value_name="y_pred")
sns.pointplot(x="Eg (eV)", y="y_pred", data=melted)


# %%
# Below is code for low-level training with tf.GradienTape. More verbose but easier to
# debug, especially with @tf.function commented out.


# %%
@tf.function
def train_step(features, bandgaps):
    with tf.GradientTape() as tape:
        preds = model(features)
        loss = loss_fn(bandgaps, preds)
        tf.summary.scalar("VI lower bound loss (NLL + KL)", loss)
    grads = tape.gradient(loss, model.trainable_variables)
    adam.apply_gradients(zip(grads, model.trainable_variables))

    mae = tf.reduce_mean(tf.metrics.mae(bandgaps, preds))
    return mae


def train():
    steps = len(y_train) // flags.batch_size  # steps per epoch
    for epoch in range(flags.epochs):
        for _ in tqdm(range(steps), desc=f"epoch {epoch + 1}/{flags.epochs}"):
            batch = X_train.sample(flags.batch_size)
            tf.summary.experimental.set_step(adam.iterations)
            mae = train_step(batch.values, y_train.loc[batch.index].values)
            tf.summary.scalar("MAE training", mae)

        # Accuracy estimated by single call for speed. Would be more accurate to
        # approximately integrate over the parameter posteriors by averaging across
        # multiple calls.
        y_val_pred = model(X_test.values)
        val_acc = tf.reduce_mean(tf.metrics.mae(y_test.values, y_val_pred))

        tf.summary.scalar("MAE validation", val_acc)
        print(f"MAE on validation set: {val_acc:.4} eV")


# %%
log_writer = tf.summary.create_file_writer(
    f"{flags.logdir}/{datetime.now():%m.%d-%H:%M:%S}"
)
log_writer.set_as_default()

train()
