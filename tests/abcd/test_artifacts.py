import json

import pandas as pd

from abcd.plot import plot
from abcd.summarize import summarize


def test_summary_and_plot_are_generated_from_completed_runs(tmp_path):
    run = tmp_path / "garden" / "abcd"
    run.mkdir(parents=True)
    (run / "run.json").write_text(
        json.dumps(
            {
                "dataset": "/datasets/garden",
                "method": "abcd",
                "seed": 0,
                "git_revision": "abc123",
            }
        )
    )
    pd.DataFrame(
        [
            {"psnr": 20.0, "ssim": 0.7, "lpips": 0.2},
            {"psnr": 22.0, "ssim": 0.8, "lpips": 0.1},
        ]
    ).to_csv(run / "evaluation.csv", index=False)
    (run / "training.jsonl").write_text(
        json.dumps(
            {
                "type": "memory_snapshot",
                "timestamp_s": 2.5,
                "ram_mb": 100,
                "vram_mb": 50,
            }
        )
        + "\n"
    )

    results_path = summarize(tmp_path)
    plot_path = plot(results_path)
    records = json.loads(results_path.read_text())

    assert records[0]["psnr"] == 21.0
    assert records[0]["ssim"] == 0.75
    assert records[0]["peak_vram_mb"] == 50
    assert results_path.with_suffix(".csv").is_file()
    assert plot_path.is_file()
