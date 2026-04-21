import os
import glob
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BENCHMARK_DIR = "./benchmark_results"
OUTPUT_DIR = "./benchmark_plots"


def load_results():
    results = {}
    pattern = re.compile(r"(.+)_(vanilla|grid_disabled|grid_last_gpu|grid_last_cpu)")

    for folder in sorted(os.listdir(BENCHMARK_DIR)):
        match = pattern.match(folder)
        if not match:
            continue
        dataset, method = match.groups()

        memory_path = os.path.join(BENCHMARK_DIR, folder, "memory.csv")
        training_path = os.path.join(BENCHMARK_DIR, folder, "training.csv")

        if not os.path.exists(memory_path) or not os.path.exists(training_path):
            continue

        memory_df = pd.read_csv(memory_path)
        training_df = pd.read_csv(training_path)

        if dataset not in results:
            results[dataset] = {}
        results[dataset][method] = {
            "memory": memory_df,
            "training": training_df,
        }

    return results


def plot_dataset(results, dataset, methods_order):
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(f"Memory Profile: {dataset}", fontsize=14, fontweight="bold")

    colors = {
        "vanilla": "#1f77b4",
        "grid_disabled": "#ff7f0e",
        "grid_last_gpu": "#2ca02c",
        "grid_last_cpu": "#d62728",
    }

    ax_ram, ax_vram = axes

    for method in methods_order:
        if method not in results:
            continue
        data = results[method]
        memory = data["memory"]
        training = data["training"]

        t = memory["timestamp_s"].values
        ram = memory["ram_mb"].values
        vram = memory["vram_mb"].values

        label = method.replace("_", " ").title()
        color = colors.get(method, "#333333")

        ax_ram.plot(t, ram / 1024, label=label, color=color, alpha=0.8, linewidth=1)
        ax_vram.plot(t, vram / 1024, label=label, color=color, alpha=0.8, linewidth=1)

    ax_ram.set_ylabel("RAM (GB)", fontsize=11)
    ax_ram.legend(loc="upper right", fontsize=9)
    ax_ram.grid(True, alpha=0.3)
    ax_ram.set_ylim(bottom=0)

    ax_vram.set_ylabel("VRAM (GB)", fontsize=11)
    ax_vram.set_xlabel("Time (s)", fontsize=11)
    ax_vram.legend(loc="upper right", fontsize=9)
    ax_vram.grid(True, alpha=0.3)
    ax_vram.set_ylim(bottom=0)

    plt.tight_layout()
    return fig


def plot_all_datasets(all_results):
    methods_order = ["vanilla", "grid_disabled", "grid_last_gpu", "grid_last_cpu"]
    datasets = sorted(all_results.keys())

    n_datasets = len(datasets)
    fig, axes = plt.subplots(n_datasets, 2, figsize=(16, 4 * n_datasets), sharex="col")

    if n_datasets == 1:
        axes = axes.reshape(1, -1)

    colors = {
        "vanilla": "#1f77b4",
        "grid_disabled": "#ff7f0e",
        "grid_last_gpu": "#2ca02c",
        "grid_last_cpu": "#d62728",
    }

    for row_idx, dataset in enumerate(datasets):
        results = all_results[dataset]
        ax_ram = axes[row_idx, 0]
        ax_vram = axes[row_idx, 1]

        for method in methods_order:
            if method not in results:
                continue
            data = results[method]
            memory = data["memory"]

            t = memory["timestamp_s"].values
            ram = memory["ram_mb"].values / 1024
            vram = memory["vram_mb"].values / 1024

            label = method.replace("_", " ").title()
            color = colors.get(method, "#333333")

            ax_ram.plot(t, ram, label=label, color=color, alpha=0.8, linewidth=1)
            ax_vram.plot(t, vram, label=label, color=color, alpha=0.8, linewidth=1)

        ax_ram.set_ylabel(f"{dataset}\nRAM (GB)", fontsize=10)
        ax_vram.set_ylabel(f"{dataset}\nVRAM (GB)", fontsize=10)
        ax_ram.grid(True, alpha=0.3)
        ax_vram.grid(True, alpha=0.3)
        ax_ram.set_ylim(bottom=0)
        ax_vram.set_ylim(bottom=0)

        if row_idx == 0:
            ax_ram.legend(loc="upper right", fontsize=8)
            ax_vram.legend(loc="upper right", fontsize=8)

    axes[-1, 0].set_xlabel("Time (s)", fontsize=11)
    axes[-1, 1].set_xlabel("Time (s)", fontsize=11)

    fig.suptitle(
        "Memory Profile Comparison Across Datasets",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    return fig


def create_summary_table(all_results):
    methods_order = ["vanilla", "grid_disabled", "grid_last_gpu", "grid_last_cpu"]

    rows = []
    for dataset in sorted(all_results.keys()):
        results = all_results[dataset]
        for method in methods_order:
            if method not in results:
                continue
            memory = results[method]["memory"]
            training = results[method]["training"]

            rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "peak_ram_gb": memory["ram_mb"].max() / 1024,
                    "peak_vram_gb": memory["vram_mb"].max() / 1024,
                    "total_time_s": memory["timestamp_s"].max(),
                    "total_iterations": training["iteration"].max() + 1,
                }
            )

    return pd.DataFrame(rows)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_results = load_results()

    if not all_results:
        print("No benchmark results found")
        return

    print(f"Loaded results for {len(all_results)} dataset(s)")
    for dataset, methods in all_results.items():
        print(f"  {dataset}: {list(methods.keys())}")

    methods_order = ["vanilla", "grid_disabled", "grid_last_gpu", "grid_last_cpu"]

    for dataset, results in all_results.items():
        fig = plot_dataset(results, dataset, methods_order)
        fig.savefig(
            os.path.join(OUTPUT_DIR, f"{dataset}_memory.png"),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(fig)
        print(f"Saved: {dataset}_memory.png")

    fig_all = plot_all_datasets(all_results)
    fig_all.savefig(
        os.path.join(OUTPUT_DIR, "all_datasets_memory.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig_all)
    print("Saved: all_datasets_memory.png")

    summary = create_summary_table(all_results)
    summary_path = os.path.join(OUTPUT_DIR, "summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"Saved: summary.csv")

    print("\nPeak Memory Summary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
