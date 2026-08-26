import json
from pathlib import Path

import matplotlib.pyplot as plt
import tyro


def plot(results: Path, output: Path | None = None) -> Path:
    """Generate quality and resource comparison plots from summarized runs."""

    records = json.loads(results.read_text(encoding="utf-8"))
    output = output or results.with_name("comparison.png")
    labels = [f"{record['scene']}\n{record['method']}" for record in records]
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    metrics = [
        ("psnr", "PSNR", True),
        ("ssim", "SSIM", True),
        ("peak_vram_mb", "Peak VRAM (MB)", False),
        ("time_s", "Time (s)", False),
    ]
    for axis, (key, title, _) in zip(axes.flat, metrics):
        values = [record[key] or 0 for record in records]
        axis.bar(range(len(records)), values)
        axis.set_title(title)
        axis.set_xticks(range(len(records)), labels, rotation=30, ha="right")
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200)
    plt.close(figure)
    return output


def main() -> None:
    tyro.cli(plot)


if __name__ == "__main__":
    main()
