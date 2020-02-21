# %%
import argparse
from datetime import datetime

import pandas as pd
import seaborn as sns
import tensorflow as tf
from tqdm import tqdm

from models import NFFeedForward

# %%
parser = argparse.ArgumentParser(allow_abbrev=False)
# TensorBoard log directory
parser.add_argument("-logdir", type=str, default="logs/bandgap/")
parser.add_argument("-epochs", type=int, default=5)
parser.add_argument("-batch_size", type=int, default=64)
parser.add_argument("-n_flows_q", type=int, default=2)
parser.add_argument("-n_flows_r", type=int, default=2)
parser.add_argument("-use_z", action="store_false")
parser.add_argument("-learn_p", action="store_true")
parser.add_argument("-seed", type=int, default=1)
parser.add_argument("-learning_rate", type=float, default=0.001)
parser.add_argument("-thres_std", type=float, default=1)
parser.add_argument("-flow_dim_h", type=int, default=50)
parser.add_argument("-test_samples", type=int, default=50)
parser.add_argument("-std_init", type=float, default=1e-2)
flags, _ = parser.parse_known_args()

# %%
features = pd.read_csv("../data/bandgaps.csv")
composition = features.pop("Composition")
features = features.astype("float32")
bandgaps = features.pop("Eg (eV)")

X_train = features.sample(frac=0.8, random_state=flags.seed)
X_test = features.drop(X_train.index)
X_val = X_test.sample(frac=0.5, random_state=flags.seed)
X_test = X_test.drop(X_val.index)
y_train, y_val, y_test = [bandgaps[X.index] for X in [X_train, X_val, X_test]]
n_samples = len(y_train)

tf.random.set_seed(flags.seed)

# %%
model = NFFeedForward(
    layer_dims=[100, 50, 10, 1],
    n_flows_q=flags.n_flows_q,
    n_flows_r=flags.n_flows_r,
    use_z=flags.use_z,
    learn_p=flags.learn_p,
    thres_std=flags.thres_std,
    flow_dim_h=flags.flow_dim_h,
    std_init=flags.std_init,
)
optimizer = tf.optimizers.Adam(learning_rate=1e-3)


# %%
def loss_fn(y_true, y_pred):
    mse = tf.reduce_mean(tf.losses.mse(y_true, y_pred))
    tf.summary.scalar("MSE", mse)

    kl_loss = model.kl_div() / n_samples
    tf.summary.scalar("KL regularization loss", kl_loss)

    return mse + kl_loss


# @tf.function
def train_step(features, bandgaps):
    with tf.GradientTape() as tape:
        preds = model(features)
        loss = loss_fn(bandgaps, preds)
        tf.summary.scalar("VI lower bound loss (NLL + KL)", loss)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    mae = tf.reduce_mean(tf.metrics.mae(bandgaps, preds))
    return mae


def train():
    steps = n_samples // flags.batch_size  # steps per epoch
    for epoch in range(flags.epochs):
        for _ in tqdm(range(steps), desc=f"epoch {epoch + 1}/{flags.epochs}",):
            batch = X_train.sample(flags.batch_size)
            tf.summary.experimental.set_step(optimizer.iterations)
            mae = train_step(batch.values, y_train.loc[batch.index].values)
            tf.summary.scalar("MAE training", mae)

        # Accuracy estimated by single call for speed. Would be more accurate to
        # approximately integrate over the parameter posteriors by averaging across
        # multiple calls.
        y_val_pred = model(X_val.values)
        val_acc = tf.reduce_mean(tf.metrics.mae(y_val.values, y_val_pred))

        tf.summary.scalar("MAE validation", val_acc)
        print(f"MAE on validation set: {val_acc:.4g} eV")


# %%
def predict(X=X_test.values, n_samples=flags.test_samples):
    preds = []
    for i in tqdm(range(n_samples), desc="Sampling"):
        preds.append(model(X))
    return tf.squeeze(preds)


# %%
log_writer = tf.summary.create_file_writer(
    flags.logdir + datetime.now().strftime("%Y.%m.%d-%H:%M:%S")
)
log_writer.set_as_default()

train()

# %%
preds = predict().numpy()
preds = pd.DataFrame(preds, columns=composition.loc[y_test.index]).T.reset_index()
preds["Eg (eV)"] = y_test.values

# %%
sns.pointplot(
    x="Eg (eV)",
    y="y_pred",
    data=preds.melt(id_vars=["Composition", "Eg (eV)"], value_name="y_pred"),
)
