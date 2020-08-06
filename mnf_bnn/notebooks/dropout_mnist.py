# %%
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
import torchvision as tv
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from mnf_bnn import ROOT

# %%
plt.rcParams["figure.figsize"] = [12, 8]

# Download MNIST dataset if not already at data_dir.
# tv.transforms.Normalize() seems to be unnecessary.
to_tensor, to_pil = tv.transforms.ToTensor(), tv.transforms.ToPILImage()
kwargs = dict(root=ROOT + "/data", transform=to_tensor, download=True)
train_set = tv.datasets.MNIST(**kwargs, train=True)
test_set = tv.datasets.MNIST(**kwargs, train=False)

train_loader = DataLoader(
    dataset=train_set, batch_size=32, shuffle=True, drop_last=True
)
test_loader = DataLoader(dataset=test_set, batch_size=32, shuffle=False, drop_last=True)


# %%
for idx in range(5):
    ax = plt.subplot(1, 5, idx + 1, title=f"ground truth: {train_set[idx][1]}")
    ax.imshow(train_set[idx][0].squeeze(), cmap="gray")
    ax.axis("off")


# %%
class LeNet5(nn.Module):
    def __init__(self, n_classes=10, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, n_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class LeNet5Dropout(LeNet5):
    def __init__(self, drop_rate=0.5, **kwargs):
        super().__init__(**kwargs)
        self.drop_rate = drop_rate

    def forward(self, x):
        # unlike nn.Dropout, functional.dropout is always on (even at test time)
        relu_drop = lambda arg, rate=1: F.relu(F.dropout(arg, self.drop_rate * rate))
        x = relu_drop(self.conv1(x), 0.5)
        x = F.max_pool2d(x, 2)
        x = relu_drop(self.conv2(x), 0.5)
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = relu_drop(self.fc1(x))
        x = relu_drop(self.fc2(x))
        x = self.fc3(x)
        return x


# %%
def train(model, loader, epochs=1, print_every=50, **kwargs):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), **kwargs)
    for epoch in range(epochs):
        # print("epoch\tbatch\t\ttraining loss")
        for batch, (samples, targets) in tqdm(
            enumerate(loader), desc=f"Epoch {epoch + 1}/{epochs}"
        ):
            optimizer.zero_grad()
            output = model(samples)
            loss = F.cross_entropy(output, targets)
            loss.backward()
            optimizer.step()
            # if batch % print_every == 0:
            #     print(f"{epoch + 1}/{epochs}\t{batch}/{len(loader)}\t\t{loss:.4g}")


def test(model, loader):
    model.eval()
    test_loss = correct = 0
    for data, targets in tqdm(loader, desc="Predicting on test set"):
        output = model(data)
        test_loss += F.cross_entropy(output, targets)
        preds = output.max(1)[1]  # [1]: grab indices (not values)
        correct += (preds == targets).sum()

    print(
        f"\n\nAvg. test loss: {test_loss / len(loader):.4g}, "
        f"Accuracy: {int(correct) / len(test_loader.dataset):.2g}"
    )


# %%
lenet = LeNet5()


# %%
train(lenet, train_loader)


# %%
test(lenet, test_loader)


# %%
lenet_dropout = LeNet5Dropout()


# %%
train(lenet_dropout, train_loader)


# %%
test(lenet_dropout, test_loader)


# %%
def dropout_test(model, inputs, n_preds=100):
    model.eval()
    output = [model(inputs) for _ in range(n_preds)]
    return torch.stack(output).squeeze()


pic, target = test_set[7]


# %%
def exit_reenter_training_manifold(pred_fn, plot_type="violin"):
    """Start with a 9 (in the training set) and rotate it in steps of 20° until 180°.
    By then it looks like a 6 (back in the training set). In between it wasn't like a
    valid digit so a good Bayesian model should assign it increased uncertainty.

    Args:
        pred_fn ([type]): [description]
    """
    for idx in range(9):
        ax1 = plt.subplot(3, 3, idx + 1)

        pic9_rot = to_tensor(tv.transforms.functional.rotate(to_pil(pic), idx * 20))

        # insert batch dimension
        preds = pred_fn(pic9_rot[None, ...])

        preds = F.softmax(preds).detach().numpy()

        df = pd.DataFrame(preds).melt(var_name="digit", value_name="softmax")
        # scale="count": set violin width according to number of predictions in that bin
        # cut=0: limit the violin range to the range of observed data
        if plot_type == "violin":
            sns.violinplot(
                data=df, x="digit", y="softmax", scale="count", cut=0, ax=ax1
            )
        elif plot_type == "bar":
            sns.barplot(data=df, x="digit", y="softmax", ax=ax1)

        ax1.set(ylim=[None, 1.1], title=f"mean max: {preds.mean(0).argmax()}")
        ax2 = ax1.inset_axes([0, 0.5, 0.4, 0.4])
        ax2.axis("off")
        ax2.imshow(pic9_rot.squeeze(), cmap="gray")

    plt.tight_layout()


# %%
exit_reenter_training_manifold(lambda x: dropout_test(lenet_dropout, x))
plt.savefig("rot9-mnf-lenet.svg", bbox_inches="tight")


# %%
exit_reenter_training_manifold(lambda x: lenet(x), plot_type="bar")
