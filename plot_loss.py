import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_loss_data(file_path):
    """
    Load loss data from .npy file.
    Supports:
    1. 1D array: [loss1, loss2, ...]
    2. 2D array: use first column by default
    3. dict saved as npy: {'loss': [...]} or {'train_loss': [...]}
    """
    data = np.load(file_path, allow_pickle=True)

    if isinstance(data.item(), dict):
        data = data.item()
        for key in ["loss", "train_loss", "train_losses", "loss_data"]:
            if key in data:
                return np.asarray(data[key], dtype=float)
        raise KeyError(
            f"Cannot find loss key in dict. Available keys: {list(data.keys())}"
        )

    data = np.asarray(data, dtype=float)

    if data.ndim == 1:
        return data
    elif data.ndim == 2:
        return data[:, 0]
    else:
        raise ValueError(f"Unsupported loss data shape: {data.shape}")


def plot_loss(loss_data, save_path="loss_curve.png"):
    epochs = np.arange(1, len(loss_data) + 1)

    # Nature-style figure settings
    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 8,
        "axes.linewidth": 0.8,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": 300,
        "savefig.dpi": 600,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, ax = plt.subplots(figsize=(5.0, 3.0))

    ax.plot(
        epochs,
        loss_data,
        linewidth=1.4,
        color="#1f77b4",
        label="Training loss"
    )

    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.tick_params(
        direction="out",
        length=3,
        width=0.8,
        colors="black"
    )

    ax.grid(
        True,
        which="major",
        linestyle="--",
        linewidth=0.5,
        alpha=0.35
    )

    ax.legend(frameon=False)

    ax.set_xlim(1, len(loss_data))

    # If all losses are positive, make lower boundary slightly below minimum
    ymin, ymax = np.min(loss_data), np.max(loss_data)
    margin = 0.08 * (ymax - ymin) if ymax > ymin else 0.1 * ymax
    ax.set_ylim(max(0, ymin - margin), ymax + margin)

    fig.tight_layout()

    save_path = Path(save_path)
    fig.savefig(save_path, bbox_inches="tight")
    # fig.savefig(save_path.with_suffix(".pdf"), bbox_inches="tight")

    plt.show()


if __name__ == "__main__":
    loss_file = "results/unet/loss_data.npy"
    loss_data = load_loss_data(loss_file)

    plot_loss(
        loss_data,
        save_path="results/unet/loss_curve.png"
    )