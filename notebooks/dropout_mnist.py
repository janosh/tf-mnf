# %%
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from tqdm import tqdm

from tf_mnf import ROOT
from tf_mnf.evaluate import np2torch2np, rot_img


# %%
plt.rcParams["figure.figsize"] = [12, 8]

# tv.transforms.Normalize() seems to be unnecessary.
train_set, test_set = (
    MNIST(root=ROOT + "/data", transform=ToTensor(), download=True, train=x)
    for x in [True, False]
)

train_loader, test_loader = (
    DataLoader(dataset=x, batch_size=32, shuffle=True, drop_last=True)
    for x in [train_set, test_set]
)


# %%
for idx in range(5):
    ax = plt.subplot(1, 5, idx + 1, title=f"label: {train_set[idx][1]}")
    ax.imshow(train_set.data[idx], cmap="gray")
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
        for _batch, (samples, targets) in tqdm(
            enumerate(loader), desc=f"Epoch {epoch + 1}/{epochs}"
        ):
            optimizer.zero_grad()
            output = model(samples)
            loss = F.cross_entropy(output, targets)
            loss.backward()
            optimizer.step()
            # if batch % print_every == 0:
            #     print(f"{epoch + 1}/{epochs}\t{batch}/{len(loader)}\t\t{loss:.4}")


def test(model, loader):
    model.eval()
    test_loss = correct = 0
    for data, targets in tqdm(loader, desc="Predicting on test set"):
        output = model(data)
        test_loss += F.cross_entropy(output, targets)
        preds = output.max(1)[1]  # [1]: grab indices (not values)
        correct += (preds == targets).sum()

    print(
        f"\n\nAvg. test loss: {test_loss / len(loader):.4}, "
        f"Accuracy: {int(correct) / len(test_loader.dataset):.2}"
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
img9 = test_set[7][0]

rot_img(np2torch2np(lambda x: F.softmax(lenet_dropout(x.repeat(50, 1, 1, 1)))), img9)
# plt.savefig("rot9-mnf-lenet.svg", bbox_inches="tight")


# %%
rot_img(np2torch2np(lambda x: F.softmax(lenet(x))), img9, plot_type="bar")
# plt.savefig("rot9-lenet.svg", bbox_inches="tight")
