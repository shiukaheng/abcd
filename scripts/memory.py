"""Measure inactive-partition VRAM scaling."""

import tyro

from abcd.memory import measure_memory

if __name__ == "__main__":
    tyro.cli(measure_memory)
