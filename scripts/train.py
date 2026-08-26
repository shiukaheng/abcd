"""Run ABCD, its ablation, or the 3DGS baseline."""

import tyro

from abcd.train import train

if __name__ == "__main__":
    tyro.cli(train)
