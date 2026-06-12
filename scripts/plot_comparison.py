#!/usr/bin/env python3
"""Overlay multiple benchmark.jsonl runs on shared axes for comparison.

Usage:
    python scripts/plot_comparison.py \
      --experiment vanilla=benchmark_results/counter_vanilla/benchmark.jsonl \
      --experiment grid_og=benchmark_results/counter_grid_og/benchmark.jsonl \
      --output-dir ./test_comparison_plots

Outputs:
    {output_dir}/loss_comparison.png
    {output_dir}/gaussians_comparison.png
"""

import json
import os

import matplotlib.pyplot as plt
import tyro


def load_experiment(jsonl_path: str):
    iters = []
    gaussians = []
    losses = []
    cells = []
    with open(jsonl_path) as f:
        for line in f:
            r = json.loads(line)
            if r.get("type") != "iteration":
                continue
            iters.append(r["iteration"])
            gaussians.append(r.get("gaussians_loaded"))
            losses.append(r.get("loss"))
            cells.append(r.get("cell"))
    return {"iterations": iters, "gaussians": gaussians, "losses": losses, "cells": cells}


def _cell_segments(iterations, cells):
    seg = []
    prev = cells[0]
    for it, cell in zip(iterations, cells):
        if cell != prev:
            yield seg
            seg = []
        seg.append(it)
        prev = cell
    if seg:
        yield seg


def _ema_smooth(values, alpha):
    smoothed = [values[0]]
    for v in values[1:]:
        smoothed.append(smoothed[-1] * (1 - alpha) + v * alpha)
    return smoothed


def _smooth_per_segment(iterations, losses, cells, alpha):
    iter_smoothed = []
    loss_smoothed = []
    for seg in _cell_segments(iterations, cells):
        idx = [iterations.index(it) for it in seg]
        seg_losses = [losses[i] for i in idx]
        seg_smooth = _ema_smooth(seg_losses, alpha)
        iter_smoothed.extend(seg)
        loss_smoothed.extend(seg_smooth)
    return iter_smoothed, loss_smoothed


def _has_switches(cells):
    if len(cells) < 2:
        return False
    return any(cells[0] != c for c in cells[1:])


def _normalize(iters):
    lo, hi = iters[0], iters[-1]
    rng = hi - lo
    if rng == 0:
        return [0.0 for _ in iters]
    return [(v - lo) / rng for v in iters]


def main(
    experiment: list[str],
    output_dir: str = "./test_comparison_plots",
    smooth_alpha: float = 0.02,
):
    os.makedirs(output_dir, exist_ok=True)

    exps = []
    for entry in experiment:
        name, path = entry.split("=", 1)
        print(f"Loading {name} from {path} ...")
        data = load_experiment(path)
        exps.append((name, data))
        print(f"  {len(data['iterations'])} iteration records")

    colors = plt.cm.tab10(range(len(exps)))

    fig, (ax_loss, ax_gauss) = plt.subplots(1, 2, figsize=(16, 6))

    for (name, data), color in zip(exps, colors):
        iters = data["iterations"]
        progress = _normalize(iters)
        gaussians = data["gaussians"]
        losses = data["losses"]
        cells = data["cells"]
        has_switch = _has_switches(cells)

        # Loss plot (log scale)
        if any(v is not None for v in losses):
            if smooth_alpha > 0 and has_switch:
                siters, sloss = _smooth_per_segment(iters, losses, cells, smooth_alpha)
                sprog = _normalize(siters)
                ax_loss.plot(progress, losses, alpha=0.1, linewidth=0.4, color=color)
                ax_loss.plot(sprog, sloss, alpha=0.9, linewidth=1.2, color=color, label=name)
            elif smooth_alpha > 0:
                sloss = _ema_smooth(losses, smooth_alpha)
                ax_loss.plot(progress, losses, alpha=0.1, linewidth=0.4, color=color)
                ax_loss.plot(progress, sloss, alpha=0.9, linewidth=1.2, color=color, label=name)
            else:
                ax_loss.plot(progress, losses, alpha=0.8, linewidth=0.8, color=color, label=name)

        # Gaussians plot
        if any(v is not None for v in gaussians):
            ax_gauss.plot(progress, gaussians, alpha=0.8, linewidth=0.8, color=color, label=name)

    ax_loss.set_xlabel("Training progress")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_yscale("log")
    ax_loss.legend(fontsize=8)
    ax_loss.set_title("Loss comparison")

    ax_gauss.set_xlabel("Training progress")
    ax_gauss.set_ylabel("Gaussians loaded")
    ax_gauss.legend(fontsize=8)
    ax_gauss.set_title("Gaussians comparison")

    plt.tight_layout()
    out = os.path.join(output_dir, "comparison.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


if __name__ == "__main__":
    tyro.cli(main)
