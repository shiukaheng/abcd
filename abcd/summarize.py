import csv
import json
from pathlib import Path

import pandas as pd
import tyro


def _resource_summary(path: Path) -> dict[str, float | None]:
    peak_ram = None
    peak_vram = None
    elapsed = None
    if not path.is_file():
        return {"peak_ram_mb": None, "peak_vram_mb": None, "time_s": None}
    for line in path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        elapsed = max(elapsed or 0.0, record.get("timestamp_s", 0.0))
        if record["type"] == "memory_snapshot":
            peak_ram = max(peak_ram or 0.0, record["ram_mb"])
            peak_vram = max(peak_vram or 0.0, record["vram_mb"])
    return {"peak_ram_mb": peak_ram, "peak_vram_mb": peak_vram, "time_s": elapsed}


def summarize(runs: Path, output: Path | None = None) -> Path:
    """Aggregate run manifests, held-out metrics, and resource logs."""

    output = output or runs / "results.json"
    records = []
    for manifest_path in sorted(runs.rglob("run.json")):
        run_dir = manifest_path.parent
        evaluation_path = run_dir / "evaluation.csv"
        if not evaluation_path.is_file():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        evaluation = pd.read_csv(evaluation_path)
        records.append(
            {
                "scene": Path(manifest["dataset"]).name,
                "method": manifest["method"],
                "seed": manifest["seed"],
                "psnr": float(evaluation["psnr"].mean()),
                "ssim": float(evaluation["ssim"].mean()),
                "lpips": float(evaluation["lpips"].mean()),
                **_resource_summary(run_dir / "training.jsonl"),
                "run": str(run_dir),
                "git_revision": manifest["git_revision"],
            }
        )

    if not records:
        raise ValueError(f"No completed runs found under {runs}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(records, indent=2, sort_keys=True), encoding="utf-8")
    with output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    return output


def main() -> None:
    tyro.cli(summarize)


if __name__ == "__main__":
    main()
