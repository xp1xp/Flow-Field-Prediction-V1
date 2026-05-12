import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable


DEFAULT_PREDICTION_PATH = Path("results/unet/real_prediction.npy")
DEFAULT_INPUT_PATH = Path("data/data_real2c.npy")
DEFAULT_OUTPUT_PATH = Path("results/unet/plot_real_predicition/real_prediction.png")

REAL_INPUT_COLOR_LIMITS = {
    'u': (50, 190),
    'v': (-55, 45),
    'w': (-30, 30),
}


def add_matched_colorbar(fig, ax, im, label):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.set_ylabel(label, rotation=270, labelpad=12)
    return cbar


def resolve_prediction_path(path):
    path = Path(path)
    if path.exists():
        return path
    raise FileNotFoundError(f"Prediction file not found: {path}")


def load_optional_input(path):
    path = Path(path)
    if not path.exists():
        return None
    data = np.load(path)
    if data.ndim != 3 or data.shape[0] < 2:
        raise ValueError(f"Expected input shape (2, H, W), got {data.shape}")
    return data


def plot_real_prediction(input_2d, prediction_3d, save_path):
    if prediction_3d.ndim != 3 or prediction_3d.shape[0] < 3:
        raise ValueError(f"Expected prediction shape (3, H, W), got {prediction_3d.shape}")

    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    velocity_components = ["u", "v", "w"]

    if input_2d is not None:
        for i, comp in enumerate(["u", "v"]):
            ax = fig.add_subplot(gs[i, 0])
            im = ax.imshow(np.rot90(input_2d[i], k=1), cmap="jet", origin="lower")
            ax.set_title(f"Input 2D {comp.upper()} Velocity")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            add_matched_colorbar(fig, ax, im, "Velocity")

    for i, comp in enumerate(velocity_components):
        vmin, vmax = REAL_INPUT_COLOR_LIMITS[comp]
        ax = fig.add_subplot(gs[i, 1])
        im = ax.imshow(np.rot90(prediction_3d[i], k=1), cmap="jet", origin="lower", vmin=vmin, vmax=vmax)
        ax.set_title(f"Predicted 3D {comp.upper()} Velocity")
        ax.set_xlabel("Z")
        ax.set_ylabel("Y")
        add_matched_colorbar(fig, ax, im, "Velocity")

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Visualization saved to {save_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Plot real single-sample 3D flow prediction.")
    parser.add_argument("--prediction", default=str(DEFAULT_PREDICTION_PATH), help="Path to real_prediction.npy")
    parser.add_argument("--input", default=str(DEFAULT_INPUT_PATH), help="Path to input 2D data_real2c.npy")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH), help="Path to save the figure")
    return parser.parse_args()


def main():
    args = parse_args()
    prediction_path = resolve_prediction_path(args.prediction)
    prediction = np.load(prediction_path)
    input_2d = load_optional_input(args.input)
    print(f"Prediction: {prediction_path} {prediction.shape}")
    if input_2d is not None:
        print(f"Input: {args.input} {input_2d.shape}")
    plot_real_prediction(input_2d, prediction, args.output)


if __name__ == "__main__":
    main()
