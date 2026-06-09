#!/usr/bin/env python3
"""Convert benchmark.jsonl to Chrome Trace Event format.

Usage:
    python scripts/jsonl_to_trace.py /tmp/test_logger/benchmark.jsonl
    # Produces /tmp/test_logger/trace.json
    # Drag it into ui.perfetto.dev or chrome://tracing

Output tracks:
  - MemoryActual:      vram_mb, ram_mb counter lines (per snapshot)
  - TheoryGPU:         counter lines for model/cameras/cells/precomp/merged bytes on GPU
  - TheoryCPU:         counter lines for model/cameras/cells/prerenders/merged bytes on CPU
  - Iterations:        instant markers at each logged iteration
"""

import json
import re
import sys
from typing import Any, Dict, List, Tuple


def _categorize(key: str) -> str:
    if key.startswith("merged."):
        return "merged"
    if key.startswith("model."):
        return "model"
    if re.match(r"^cam_\d+\.image$", key):
        return "cameras"
    if re.match(r"^cam_\d+\.precomp_(bg|fg)$", key):
        return "precomp"
    if re.match(r"^cell_.+\.prerender\.", key):
        return "prerenders"
    if re.match(r"^cell_.+\.(positions|sh_coefficients|rotations|scales|opacities)$", key):
        return "cells"
    return "other"


def _classify_device(device: str) -> str:
    if "cuda" in device or "gpu" in device:
        return "gpu"
    return "cpu"


def _device_for_category(category: str) -> Dict[str, float]:
    return {"model": 0.0, "cameras": 0.0, "cells": 0.0,
            "precomp": 0.0, "prerenders": 0.0, "merged": 0.0, "other": 0.0}


def convert(input_path: str, output_path: str) -> None:
    events: List[Dict[str, Any]] = []

    live: Dict[str, Tuple[int, str]] = {}

    with open(input_path) as f:
        lines = f.readlines()

    # --- pass 1: replay events to collect memory_snapshot timestamps ---
    snapshot_timestamps = []
    for line in lines:
        record = json.loads(line)
        if record.get("type") == "memory_snapshot":
            snapshot_timestamps.append(record["timestamp_s"])

    # --- pass 2: replay events, accumulate state, emit counters at snapshot times ---
    snapshot_idx = 0
    for line in lines:
        record = json.loads(line)

        if record.get("type") == "memory_snapshot":
            ts_us = int(record["timestamp_s"] * 1_000_000 if snapshot_idx < len(snapshot_timestamps) else 0)

            events.append({
                "ph": "C",
                "name": "MemoryActual",
                "pid": 0,
                "tid": 0,
                "ts": ts_us,
                "args": {
                    "vram_mb": round(record["vram_mb"], 2),
                    "ram_mb": round(record["ram_mb"], 2),
                },
            })

            gpu = _device_for_category("")
            cpu = _device_for_category("")
            for key, (b, dev) in live.items():
                cat = _categorize(key)
                mb = b / (1024 * 1024)
                if dev == "gpu":
                    gpu[cat] += mb
                else:
                    cpu[cat] += mb

            total_gpu = sum(gpu.values())
            total_cpu = sum(cpu.values())

            events.append({
                "ph": "C",
                "name": "TheoryGPU",
                "pid": 0,
                "tid": 0,
                "ts": ts_us,
                "args": {
                    "model_mb": round(gpu["model"], 2),
                    "cameras_mb": round(gpu["cameras"], 2),
                    "cells_mb": round(gpu["cells"], 2),
                    "precomp_mb": round(gpu["precomp"], 2),
                    "merged_mb": round(gpu["merged"], 2),
                    "other_mb": round(gpu["other"], 2),
                    "total_mb": round(total_gpu, 2),
                },
            })

            events.append({
                "ph": "C",
                "name": "TheoryCPU",
                "pid": 0,
                "tid": 0,
                "ts": ts_us,
                "args": {
                    "model_mb": round(cpu["model"], 2),
                    "cameras_mb": round(cpu["cameras"], 2),
                    "cells_mb": round(cpu["cells"], 2),
                    "prerenders_mb": round(cpu["prerenders"], 2),
                    "merged_mb": round(cpu["merged"], 2),
                    "other_mb": round(cpu["other"], 2),
                    "total_mb": round(total_cpu, 2),
                },
            })

            snapshot_idx += 1

        elif record.get("type") == "iteration":
            ts_us = int(record["timestamp_s"] * 1_000_000)
            events.append({
                "ph": "i",
                "name": f"iteration {record['iteration']}",
                "pid": 0,
                "tid": 0,
                "ts": ts_us,
                "s": "g",
                "args": {"iteration": record["iteration"]},
            })

        elif record.get("message") == "tensor":
            dev = _classify_device(record.get("device", ""))
            if record.get("event") == "set":
                live[record["key"]] = (record["bytes"], dev)
            elif record.get("event") == "delete":
                live.pop(record["key"], None)

    with open(output_path, "w") as f:
        json.dump({"traceEvents": events}, f)

    lens = len(live)
    total_mb = sum(b / (1024 * 1024) for b, _ in live.values())
    print(f"Wrote {len(events)} events to {output_path}")
    print(f"Final live tensors: {lens}, total {total_mb:.1f} MB")
    if total_mb > 0:
        gpu_mb = sum(b / (1024 * 1024) for k, (b, d) in live.items() if d == "gpu")
        cpu_mb = sum(b / (1024 * 1024) for k, (b, d) in live.items() if d == "cpu")
        print(f"  GPU: {gpu_mb:.1f} MB, CPU: {cpu_mb:.1f} MB")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/jsonl_to_trace.py <benchmark.jsonl> [-o output.json]")
        sys.exit(1)

    in_path = sys.argv[1]
    out_path = in_path.replace(".jsonl", "_trace.json")
    for i, arg in enumerate(sys.argv):
        if arg == "-o" and i + 1 < len(sys.argv):
            out_path = sys.argv[i + 1]
            break

    convert(in_path, out_path)
