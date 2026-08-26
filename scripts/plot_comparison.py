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

LEGEND_NAMES = {
    "grid_smallb": "ABCD",
    "grid_smallb_no_comp": "ABCD (compositing ablated)",
    "vanilla": "3DGS",
}


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
    figsize = (4.5, 3.0)

    # Loss plot
    fig, ax = plt.subplots(figsize=figsize)
    for (name, data), color in zip(exps, colors):
        label = LEGEND_NAMES.get(name, name)
        iters = data["iterations"]
        progress = _normalize(iters)
        losses = data["losses"]
        cells = data["cells"]
        has_switch = _has_switches(cells)
        if any(v is not None for v in losses):
            if smooth_alpha > 0 and has_switch:
                siters, sloss = _smooth_per_segment(iters, losses, cells, smooth_alpha)
                sprog = _normalize(siters)
                ax.plot(progress, losses, alpha=0.1, linewidth=0.4, color=color)
                ax.plot(sprog, sloss, alpha=0.9, linewidth=1.2, color=color, label=label)
            elif smooth_alpha > 0:
                sloss = _ema_smooth(losses, smooth_alpha)
                ax.plot(progress, losses, alpha=0.1, linewidth=0.4, color=color)
                ax.plot(progress, sloss, alpha=0.9, linewidth=1.2, color=color, label=label)
            else:
                ax.plot(progress, losses, alpha=0.8, linewidth=0.8, color=color, label=label)
    ax.set_xlabel("Training progress")
    ax.set_ylabel("Loss")
    ax.set_yscale("log")
    ax.legend(fontsize=6)
    plt.tight_layout()
    out = os.path.join(output_dir, "loss_comparison.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")

    # Gaussians plot
    fig, ax = plt.subplots(figsize=figsize)
    for (name, data), color in zip(exps, colors):
        label = LEGEND_NAMES.get(name, name)
        iters = data["iterations"]
        progress = _normalize(iters)
        gaussians = data["gaussians"]
        if any(v is not None for v in gaussians):
            ax.plot(progress, gaussians, alpha=0.8, linewidth=0.8, color=color, label=label)
    ax.set_xlabel("Training progress")
    ax.set_ylabel("Gaussians loaded")
    ax.legend(fontsize=6)
    plt.tight_layout()
    out = os.path.join(output_dir, "gaussians_comparison.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


if __name__ == "__main__":
    tyro.cli(main)
