from __future__ import annotations

from functools import wraps
from typing import TYPE_CHECKING, Callable

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from scipy.ndimage import rotate


if TYPE_CHECKING:
    import numpy as np


def rot_img(
    pred_fn: Callable,
    img: np.ndarray,
    plot_type: str = "violin",
    axes: tuple[int, int] = (1, 2),
) -> None:
    """Rotate an image 180° in steps of 20°. For the example of an MNIST 9
    digit, this starts out on the training manifold, leaves it when the 9
    lies on its side and reenters it once we're at 180° and the 9 looks like
    a 6. In the middle, it's an invalid digit so a good Bayesian model should
    assign it increased uncertainty.
    """
    for idx in range(9):
        ax1 = plt.subplot(3, 3, idx + 1)

        img_rot = rotate(img, idx * 20, reshape=False, axes=axes)

        # insert batch dim
        preds = pred_fn(img_rot[None, ...])

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
        ax2 = ax1.inset_axes((0, 0.5, 0.4, 0.4))
        ax2.axis("off")
        ax2.imshow(img_rot.squeeze(), cmap="gray")

    plt.tight_layout()  # needed to keep titles clear of above subplots


def np2torch2np(func: Callable) -> Callable:
    """Convert numpy array to pytorch tensor, execute function and revert to numpy."""

    @wraps(func)
    def wrapped(x: np.ndarray) -> np.ndarray:
        return func(torch.from_numpy(x)).detach().numpy()

    return wrapped
