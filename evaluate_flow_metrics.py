import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


# 三个输出通道名称：本项目的预测结果为三分量速度场 (u, v, w)。
COMPONENT_NAMES = ("u", "v", "w")

# Nature Publishing Group (NPG) 常用配色，适合论文图中不同指标/曲线的区分。
NPG_COLORS = {
    "red": "#E64B35",
    "blue": "#4DBBD5",
    "green": "#00A087",
    "navy": "#3C5488",
    "orange": "#F39B7F",
    "purple": "#8491B4",
    "mint": "#91D1C2",
    "dark_red": "#DC0000",
    "brown": "#7E6148",
    "sand": "#B09C85",
}


def set_nature_style():
    """设置 Nature 风格的绘图参数：字体、字号、线宽、坐标轴和 600 dpi 输出。"""
    plt.rcParams.update(
        {
            "font.family": ["Arial", "Microsoft YaHei", "SimHei"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "figure.titlesize": 9,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "lines.linewidth": 1.2,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.unicode_minus": False,
        }
    )


def load_arrays(result_dir):
    """读取验证集结果文件，并检查 predictions 和 targets 的形状是否匹配。"""
    result_dir = Path(result_dir)
    arrays = {
        "input_raws": np.load(result_dir / "input_raws.npy"),
        "inputs_normalized": np.load(result_dir / "inputs_normalized.npy"),
        "predictions": np.load(result_dir / "predictions.npy"),
        "targets": np.load(result_dir / "targets.npy"),
    }
    if arrays["predictions"].shape != arrays["targets"].shape:
        raise ValueError(
            "predictions.npy and targets.npy must have the same shape: "
            f"{arrays['predictions'].shape} vs {arrays['targets'].shape}"
        )
    if arrays["predictions"].ndim != 4:
        raise ValueError(
            "Expected predictions/targets shape (samples, components, height, width), "
            f"got {arrays['predictions'].shape}"
        )
    return arrays


def safe_relative_l2(error, target, eps=1e-12):
    """计算相对 L2 误差：||pred - target||_2 / ||target||_2。"""
    return float(np.linalg.norm(error.ravel()) / (np.linalg.norm(target.ravel()) + eps))


def safe_r2(pred, target, eps=1e-12):
    """计算决定系数 R2，用于衡量预测场对目标场总体变化的解释程度。"""
    residual = np.sum((pred - target) ** 2)
    total = np.sum((target - np.mean(target)) ** 2)
    return float(1.0 - residual / (total + eps))


def safe_corr(pred, target, eps=1e-12):
    """计算 Pearson 相关系数，用于衡量预测场和目标场的线性一致性。"""
    pred_flat = pred.ravel()
    target_flat = target.ravel()
    if np.std(pred_flat) < eps or np.std(target_flat) < eps:
        return float("nan")
    return float(np.corrcoef(pred_flat, target_flat)[0, 1])


def basic_metrics(pred, target):
    """计算一组通用误差指标：MSE、RMSE、MAE、最大误差、偏差、相对 L2、R2 和相关系数。"""
    error = pred - target
    abs_error = np.abs(error)
    return {
        "mse": float(np.mean(error**2)),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(abs_error)),
        "max_ae": float(np.max(abs_error)),
        "bias": float(np.mean(error)),
        "relative_l2": safe_relative_l2(error, target),
        "r2": safe_r2(pred, target),
        "corr": safe_corr(pred, target),
    }


def compute_metrics(pred, target):
    """汇总所有评价指标，包括整体误差、分量误差、速度模长误差、分区域误差和梯度误差。"""
    error = pred - target

    # 矢量误差：每个网格点上三个速度分量误差向量的模长。
    vector_error = np.sqrt(np.sum(error**2, axis=1))

    # 速度模长误差：比较 |V_pred| 和 |V_target|，用于分析速度强弱分布是否准确。
    pred_mag = np.sqrt(np.sum(pred**2, axis=1))
    target_mag = np.sqrt(np.sum(target**2, axis=1))
    mag_error = pred_mag - target_mag

    metrics = {
        "overall": basic_metrics(pred, target),
        "components": {},
        "magnitude": basic_metrics(pred_mag, target_mag),
        "per_sample": [],
        "region": {},
        "gradient": {},
    }

    # 分通道计算 u/v/w 三个速度分量的误差。
    for index, name in enumerate(COMPONENT_NAMES[: pred.shape[1]]):
        metrics["components"][name] = basic_metrics(pred[:, index], target[:, index])

    # 逐样本误差：用于判断模型在不同验证样本上的稳定性，并寻找最差样本。
    sample_rmse = np.sqrt(np.mean(error**2, axis=(1, 2, 3)))
    sample_mae = np.mean(np.abs(error), axis=(1, 2, 3))
    sample_rel_l2 = [
        safe_relative_l2(error[i], target[i]) for i in range(pred.shape[0])
    ]
    for i in range(pred.shape[0]):
        metrics["per_sample"].append(
            {
                "sample": int(i),
                "rmse": float(sample_rmse[i]),
                "mae": float(sample_mae[i]),
                "relative_l2": float(sample_rel_l2[i]),
            }
        )

    # 按目标速度模长划分低速区、中速区和高速区，避免全局平均误差掩盖高速核心区域表现。
    percentiles = {
        "low_speed": (0, 50),
        "medium_speed": (50, 90),
        "high_speed": (90, 100),
    }
    for region, (lo, hi) in percentiles.items():
        lo_value = np.percentile(target_mag, lo)
        hi_value = np.percentile(target_mag, hi)
        if hi == 100:
            mask = target_mag >= lo_value
        else:
            mask = (target_mag >= lo_value) & (target_mag < hi_value)
        masked_vector_error = vector_error[mask]
        masked_mag_error = mag_error[mask]
        metrics["region"][region] = {
            "target_speed_percentile_low": float(lo),
            "target_speed_percentile_high": float(hi),
            "target_speed_low": float(lo_value),
            "target_speed_high": float(hi_value),
            "point_count": int(np.sum(mask)),
            "vector_mae": float(np.mean(masked_vector_error)),
            "vector_rmse": float(np.sqrt(np.mean(masked_vector_error**2))),
            "magnitude_mae": float(np.mean(np.abs(masked_mag_error))),
            "magnitude_rmse": float(np.sqrt(np.mean(masked_mag_error**2))),
        }

    # 梯度误差：比较相邻网格点差分，用于衡量模型是否保留局部空间结构和剪切变化。
    grad_pred_x = np.diff(pred, axis=3)
    grad_target_x = np.diff(target, axis=3)
    grad_pred_y = np.diff(pred, axis=2)
    grad_target_y = np.diff(target, axis=2)
    metrics["gradient"] = {
        "grad_x_rmse": float(np.sqrt(np.mean((grad_pred_x - grad_target_x) ** 2))),
        "grad_y_rmse": float(np.sqrt(np.mean((grad_pred_y - grad_target_y) ** 2))),
        "grad_x_mae": float(np.mean(np.abs(grad_pred_x - grad_target_x))),
        "grad_y_mae": float(np.mean(np.abs(grad_pred_y - grad_target_y))),
    }

    return metrics


def write_summary_csv(metrics, output_dir):
    """输出整体、速度模长和各速度分量的指标表：metric_summary.csv。"""
    rows = []
    for scope, values in [("overall", metrics["overall"]), ("magnitude", metrics["magnitude"])]:
        row = {"scope": scope, "component": "-"}
        row.update(values)
        rows.append(row)

    for component, values in metrics["components"].items():
        row = {"scope": "component", "component": component}
        row.update(values)
        rows.append(row)

    path = output_dir / "metric_summary.csv"
    fieldnames = ["scope", "component", "mse", "rmse", "mae", "max_ae", "bias", "relative_l2", "r2", "corr"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_per_sample_csv(metrics, output_dir):
    """输出每个验证样本的误差表：metric_per_sample.csv。"""
    path = output_dir / "metric_per_sample.csv"
    fieldnames = ["sample", "rmse", "mae", "relative_l2"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics["per_sample"])


def write_region_csv(metrics, output_dir):
    """输出低速/中速/高速区域的误差表：metric_region.csv。"""
    path = output_dir / "metric_region.csv"
    fieldnames = [
        "region",
        "target_speed_percentile_low",
        "target_speed_percentile_high",
        "target_speed_low",
        "target_speed_high",
        "point_count",
        "vector_mae",
        "vector_rmse",
        "magnitude_mae",
        "magnitude_rmse",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for region, values in metrics["region"].items():
            row = {"region": region}
            row.update(values)
            writer.writerow(row)


def save_json(metrics, output_dir):
    """保存完整指标字典，便于后续调试或二次分析：metrics.json。"""
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)


def save_figure(fig, output_dir, stem, formats):
    """按指定格式保存图像。默认只保存 png；需要 pdf/tiff 时可在命令行添加。"""
    for fmt in formats:
        fig.savefig(output_dir / f"{stem}.{fmt}", dpi=600)
    plt.close(fig)


def plot_component_metrics(metrics, output_dir, formats):
    """图1：绘制 u/v/w 三个速度分量的 RMSE 和 MAE 柱状图。"""
    components = list(metrics["components"].keys())
    x = np.arange(len(components))
    width = 0.36
    rmse = [metrics["components"][c]["rmse"] for c in components]
    mae = [metrics["components"][c]["mae"] for c in components]

    fig, ax = plt.subplots(figsize=(3.35, 2.75))
    ax.bar(x - width / 2, rmse, width, color=NPG_COLORS["navy"], label="RMSE")
    ax.bar(x + width / 2, mae, width, color=NPG_COLORS["red"], label="MAE")
    ax.set_xlabel("Velocity component")
    ax.set_ylabel("Error magnitude")
    ax.set_xticks(x)
    ax.set_xticklabels([c.upper() for c in components])
    ax.legend(frameon=False, ncol=2, loc="upper right")
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.5, alpha=0.8)
    save_figure(fig, output_dir, "fig_component_error_metrics", formats)


def plot_per_sample(metrics, output_dir, formats):
    """图2：绘制每个验证样本的 RMSE 和 MAE 折线图，观察样本间稳定性。"""
    samples = np.array([item["sample"] for item in metrics["per_sample"]])
    rmse = np.array([item["rmse"] for item in metrics["per_sample"]])
    mae = np.array([item["mae"] for item in metrics["per_sample"]])

    fig, ax = plt.subplots(figsize=(3.35, 2.6))
    ax.plot(samples, rmse, marker="o", color=NPG_COLORS["navy"], label="RMSE")
    ax.plot(samples, mae, marker="s", color=NPG_COLORS["green"], label="MAE")
    ax.set_xlabel("Validation sample index")
    ax.set_ylabel("Error magnitude")
    ax.set_xticks(samples)
    ax.legend(
        frameon=True,
        loc="upper right",
        ncol=2,
        handlelength=1.4,
        columnspacing=1.0,
        borderpad=0.25,
        handletextpad=0.4,
        labelspacing=0.25,
    )
    legend = ax.get_legend()
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_alpha(0.85)
    legend.get_frame().set_edgecolor("none")
    ax.grid(color="#D9D9D9", linewidth=0.5, alpha=0.8)
    save_figure(fig, output_dir, "fig_per_sample_error", formats)


def plot_region_metrics(metrics, output_dir, formats):
    """图3：绘制低速区、中速区和高速区的矢量误差/速度模长误差。"""
    regions = list(metrics["region"].keys())
    labels = ["Low", "Medium", "High"]
    x = np.arange(len(regions))
    width = 0.36
    vector_rmse = [metrics["region"][r]["vector_rmse"] for r in regions]
    magnitude_rmse = [metrics["region"][r]["magnitude_rmse"] for r in regions]

    fig, ax = plt.subplots(figsize=(3.35, 2.35))
    ax.bar(x - width / 2, vector_rmse, width, color=NPG_COLORS["blue"], label="Vector RMSE")
    ax.bar(x + width / 2, magnitude_rmse, width, color=NPG_COLORS["orange"], label="Speed RMSE")
    ax.set_xlabel("Target speed region")
    ax.set_ylabel("RMSE")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.14),
        ncol=2,
        handlelength=2.0,
        columnspacing=1.0,
    )
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.5, alpha=0.8)
    fig.subplots_adjust(top=0.78)
    save_figure(fig, output_dir, "fig_speed_region_error", formats)


def plot_parity(pred, target, output_dir, formats, max_points=30000):
    """图4：绘制预测值-真实值散点图，越接近 y=x 表示预测越准确。"""
    rng = np.random.default_rng(7)
    pred_flat = pred.ravel()
    target_flat = target.ravel()
    count = pred_flat.size
    if count > max_points:
        ids = rng.choice(count, size=max_points, replace=False)
        pred_flat = pred_flat[ids]
        target_flat = target_flat[ids]

    low = min(np.min(pred_flat), np.min(target_flat))
    high = max(np.max(pred_flat), np.max(target_flat))

    fig, ax = plt.subplots(figsize=(3.35, 3.0))
    ax.scatter(
        target_flat,
        pred_flat,
        s=2,
        color=NPG_COLORS["navy"],
        alpha=0.18,
        linewidths=0,
        rasterized=True,
    )
    ax.plot([low, high], [low, high], color=NPG_COLORS["red"], linewidth=1.0)
    ax.set_xlabel("真实速度")
    ax.set_ylabel("预测速度")
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(color="#D9D9D9", linewidth=0.5, alpha=0.8)
    save_figure(fig, output_dir, "fig_prediction_parity", formats)


def plot_error_distribution(pred, target, output_dir, formats):
    """图5：绘制所有网格点误差的概率密度分布，观察误差是否集中在 0 附近。"""
    error = (pred - target).ravel()

    fig, ax = plt.subplots(figsize=(3.35, 2.35))
    ax.hist(error, bins=80, density=True, color=NPG_COLORS["purple"], alpha=0.88)
    ax.axvline(0, color=NPG_COLORS["red"], linewidth=1.0)
    ax.set_xlabel("预测误差")
    ax.set_ylabel("Probability density")
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.5, alpha=0.8)
    save_figure(fig, output_dir, "fig_error_distribution", formats)


def add_panel_label(ax, label):
    """给多子图添加论文常用的 a/b/c 面板标号。"""
    ax.text(
        -0.08,
        1.04,
        label,
        transform=ax.transAxes,
        fontsize=8,
        fontweight="bold",
        va="bottom",
        ha="right",
    )


def rotate_map_clockwise(image):
    return np.rot90(image, k=1)#顺时针旋转函数


def get_map_sample_id(metrics, sample_index=None):
    per_sample_rmse = np.array([item["rmse"] for item in metrics["per_sample"]])
    if sample_index is None:
        return int(np.argmin(per_sample_rmse)), "best"
    sample_id = int(sample_index)
    if sample_id < 0 or sample_id >= len(per_sample_rmse):
        raise ValueError(f"sample_index must be in [0, {len(per_sample_rmse) - 1}], got {sample_id}")
    return sample_id, f"sample_{sample_id}"


def plot_sample_maps(pred, target, metrics, output_dir, formats, sample_index=None):
    """图6：绘制指定样本或 RMSE 最小样本的 u/v/w 目标场、预测场和绝对误差云图。"""
    per_sample_rmse = np.array([item["rmse"] for item in metrics["per_sample"]])
    sample_id, sample_tag = get_map_sample_id(metrics, sample_index)
    pred_sample = pred[sample_id]
    target_sample = target[sample_id]
    abs_error = np.abs(pred_sample - target_sample)

    n_components = min(pred.shape[1], 3)
    fig, axes = plt.subplots(n_components, 3, figsize=(7.1, 5.6), constrained_layout=True)
    if n_components == 1:
        axes = axes[None, :]

    panel_labels = list("abcdefghi")
    panel_index = 0
    for row in range(n_components):
        vmin = min(np.min(pred_sample[row]), np.min(target_sample[row]))
        vmax = max(np.max(pred_sample[row]), np.max(target_sample[row]))
        err_max = np.percentile(abs_error[row], 99.5)

        images = [
            (target_sample[row], "真实值", "jet", vmin, vmax),
            (pred_sample[row], "预测值", "jet", vmin, vmax),
            (abs_error[row], "绝对误差", "magma", 0, err_max),
        ]
        for col, (image, title, cmap, cmin, cmax) in enumerate(images):
            ax = axes[row, col]
            im = ax.imshow(rotate_map_clockwise(image), origin="lower", cmap=cmap, vmin=cmin, vmax=cmax, aspect="equal")
            ax.set_title(f"{title}, {COMPONENT_NAMES[row].upper()}")
            ax.set_xlabel("Z")
            ax.set_ylabel("Y")
            add_panel_label(ax, panel_labels[panel_index])
            panel_index += 1
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, shrink=0.78)
            cbar.ax.set_ylabel("速度" if col < 2 else "绝对误差", rotation=270, labelpad=9)

    fig.suptitle(f"Validation sample: index {sample_id}, RMSE = {per_sample_rmse[sample_id]:.3f}")
    save_figure(fig, output_dir, f"fig_{sample_tag}_field_maps", formats)


def plot_sample_input_maps(input_raws, metrics, output_dir, formats, sample_index=None):
    """绘制与结果云图同一样本的输入 u/v 云图。"""
    per_sample_rmse = np.array([item["rmse"] for item in metrics["per_sample"]])
    sample_id, sample_tag = get_map_sample_id(metrics, sample_index)
    input_sample = input_raws[sample_id]

    n_components = min(input_sample.shape[0], 2)
    fig, axes = plt.subplots(n_components, 1, figsize=(2.5, 3.7), constrained_layout=True)
    if n_components == 1:
        axes = np.array([axes])

    panel_labels = list("ab")
    for row in range(n_components):
        image = input_sample[row]
        vmin = np.min(image)
        vmax = np.max(image)

        ax = axes[row]
        im = ax.imshow(rotate_map_clockwise(image), origin="lower", cmap="jet", vmin=vmin, vmax=vmax, aspect="equal")
        ax.set_title(f"输入值, {COMPONENT_NAMES[row].upper()}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        add_panel_label(ax, panel_labels[row])
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, shrink=0.78)
        cbar.ax.set_ylabel("速度", rotation=270, labelpad=9)

    fig.suptitle(f"Input sample: index {sample_id}, RMSE = {per_sample_rmse[sample_id]:.3f}")
    save_figure(fig, output_dir, f"fig_{sample_tag}_input_maps", formats)


def plot_sample_speed_magnitude_map(pred, target, metrics, output_dir, formats, sample_index=None):
    """图7：绘制指定样本或 RMSE 最小样本的速度模长 |V| 目标图、预测图和绝对误差图。"""
    per_sample_rmse = np.array([item["rmse"] for item in metrics["per_sample"]])
    sample_id, sample_tag = get_map_sample_id(metrics, sample_index)
    pred_mag = np.sqrt(np.sum(pred[sample_id] ** 2, axis=0))
    target_mag = np.sqrt(np.sum(target[sample_id] ** 2, axis=0))
    abs_error = np.abs(pred_mag - target_mag)

    vmin = min(np.min(pred_mag), np.min(target_mag))
    vmax = max(np.max(pred_mag), np.max(target_mag))
    err_max = np.percentile(abs_error, 99.5)

    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.25), constrained_layout=True)
    panels = [
        (target_mag, "真实值速度", "jet", vmin, vmax),
        (pred_mag, "预测值速度", "jet", vmin, vmax),
        (abs_error, "速度绝对误差", "magma", 0, err_max),
    ]
    for i, (image, title, cmap, cmin, cmax) in enumerate(panels):
        ax = axes[i]
        im = ax.imshow(rotate_map_clockwise(image), origin="lower", cmap=cmap, vmin=cmin, vmax=cmax, aspect="equal")
        ax.set_title(title)
        ax.set_xlabel("Z")
        ax.set_ylabel("Y")
        add_panel_label(ax, "abc"[i])
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, shrink=0.78)
        cbar.ax.set_ylabel("速度" if i < 2 else "绝对误差", rotation=270, labelpad=9)

    save_figure(fig, output_dir, f"fig_{sample_tag}_speed_map", formats)


def print_key_metrics(metrics):
    """在终端打印最核心的指标，方便快速检查程序是否正常运行。"""
    overall = metrics["overall"]
    magnitude = metrics["magnitude"]
    gradient = metrics["gradient"]
    print("Overall metrics")
    print(f"  MSE        : {overall['mse']:.6f}")
    print(f"  RMSE       : {overall['rmse']:.6f}")
    print(f"  MAE        : {overall['mae']:.6f}")
    print(f"  MaxAE      : {overall['max_ae']:.6f}")
    print(f"  Relative L2: {overall['relative_l2']:.6f}")
    print(f"  R2         : {overall['r2']:.6f}")
    print("Velocity magnitude")
    print(f"  RMSE       : {magnitude['rmse']:.6f}")
    print(f"  MAE        : {magnitude['mae']:.6f}")
    print(f"  Relative L2: {magnitude['relative_l2']:.6f}")
    print("Gradient error")
    print(f"  d/dx RMSE  : {gradient['grad_x_rmse']:.6f}")
    print(f"  d/dy RMSE  : {gradient['grad_y_rmse']:.6f}")


def parse_args():#参数定义区
    """命令行参数：输入结果目录、输出目录和图片导出格式。"""
    parser = argparse.ArgumentParser(
        description="Compute flow-field prediction metrics and generate publication-quality figures."
    )
    parser.add_argument(
        "--result-dir",
        default="results/unet/evaluation_results",
        help="Directory containing input_raws.npy, inputs_normalized.npy, predictions.npy, and targets.npy.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/unet/evaluation_metrics",
        help="Directory for metric tables and figures.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        # 默认先只导出 600 dpi PNG，避免调试阶段生成体积很大的 PDF/TIFF。
        # 需要论文投稿矢量图或 TIFF 时，可命令行使用：
        # python evaluate_flow_metrics.py --formats png pdf tiff
        default=["png"],
        choices=["png", "pdf", "tiff", "svg"],
        help="Figure formats to export.",
    )
    parser.add_argument(#定义哪一张云图index（0,9）
        "--sample-index",
        type=int,
        default=1,
        help="Sample index for field-map figures. If omitted, the lowest-RMSE sample is used.",
    )
    return parser.parse_args()


def main():
    """主流程：读取数据 -> 计算指标 -> 保存表格 -> 绘制论文图 -> 打印关键结果。"""
    args = parse_args()
    set_nature_style()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 读取 predictions.npy 和 targets.npy 等验证集结果。
    arrays = load_arrays(args.result_dir)
    input_raws = arrays["input_raws"].astype(np.float64)
    pred = arrays["predictions"].astype(np.float64)
    target = arrays["targets"].astype(np.float64)

    # 2. 计算全部评价指标，并保存为 json/csv 表格。
    metrics = compute_metrics(pred, target)
    save_json(metrics, output_dir)
    write_summary_csv(metrics, output_dir)
    write_per_sample_csv(metrics, output_dir)
    write_region_csv(metrics, output_dir)

    # 3. 生成论文图。每个函数对应一张图，文件名以 fig_ 开头。
    plot_component_metrics(metrics, output_dir, args.formats)
    plot_per_sample(metrics, output_dir, args.formats)
    plot_region_metrics(metrics, output_dir, args.formats)
    plot_parity(pred, target, output_dir, args.formats)
    plot_error_distribution(pred, target, output_dir, args.formats)
    plot_sample_input_maps(input_raws, metrics, output_dir, args.formats, args.sample_index)
    plot_sample_maps(pred, target, metrics, output_dir, args.formats, args.sample_index)
    plot_sample_speed_magnitude_map(pred, target, metrics, output_dir, args.formats, args.sample_index)

    print_key_metrics(metrics)
    print(f"\nSaved metric tables and figures to: {output_dir}")


if __name__ == "__main__":
    main()
