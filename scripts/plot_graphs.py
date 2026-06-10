#!/usr/bin/env python3
"""Plot Gaussians and loss over iterations from a benchmark.jsonl file.

Usage:
    python scripts/plot_graphs.py --jsonl-path benchmark.jsonl --output-dir ./plots

Outputs:
    {output_dir}/gaussians.png   – gaussians_loaded and gaussians_total over iterations
    {output_dir}/loss.png        – loss over iterations with cell-switch markers
"""

import json
import itertools
import os

import matplotlib.pyplot as plt
import tyro


def _add_cell_switch_lines(ax, iterations, cells):
    switch_iters = []
    prev = cells[0]
    for it, cell in zip(iterations, cells):
        if cell != prev:
            switch_iters.append(it)
        prev = cell
    if not switch_iters:
        return
    for it in switch_iters:
        ax.axvline(it, color="gray", linestyle="--", alpha=0.35, linewidth=0.8)
    y0, y1 = ax.get_ylim()
    mid = (y0 + y1) / 2
    x_mid = switch_iters[len(switch_iters) // 2]
    ax.text(x_mid, y1 * 0.99, "← cell switch", ha="center", va="top",
            fontsize=7, color="gray", alpha=0.55)


def main(jsonl_path: str, output_dir: str = "./plots"):
    os.makedirs(output_dir, exist_ok=True)

    iterations = []
    gaussians_loaded = []
    gaussians_total = []
    losses = []
    cells = []

    with open(jsonl_path) as f:
        for line in f:
            record = json.loads(line)
            if record.get("type") != "iteration":
                continue
            iterations.append(record["iteration"])
            gaussians_loaded.append(record.get("gaussians_loaded"))
            gaussians_total.append(record.get("gaussians_total"))
            losses.append(record.get("loss"))
            cells.append(record.get("cell"))

    if not iterations:
        print("No iteration events found in", jsonl_path)
        return

    has_switches = any(cells[0] != c for c in itertools.islice(cells, 1, None))

    plt.figure(figsize=(10, 5))
    ax = plt.gca()
    has_loaded = any(v is not None for v in gaussians_loaded)
    has_total = any(v is not None for v in gaussians_total)
    if has_loaded:
        ax.plot(iterations, gaussians_loaded, label="gaussians_loaded", alpha=0.8)
    if has_total:
        ax.plot(iterations, gaussians_total, label="gaussians_total", alpha=0.8)
    if has_switches:
        _add_cell_switch_lines(ax, iterations, cells)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Gaussians")
    if has_loaded or has_total:
        ax.legend()
    plt.tight_layout()
    out_gauss = os.path.join(output_dir, "gaussians.png")
    plt.savefig(out_gauss, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_gauss}")

    if any(v is not None for v in losses):
        plt.figure(figsize=(10, 5))
        ax = plt.gca()
        ax.plot(iterations, losses, alpha=0.8, linewidth=0.8)
        if has_switches:
            _add_cell_switch_lines(ax, iterations, cells)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Loss")
        ax.set_yscale("log")
        plt.tight_layout()
        out_loss = os.path.join(output_dir, "loss.png")
        plt.savefig(out_loss, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {out_loss}")


if __name__ == "__main__":
    tyro.cli(main)
