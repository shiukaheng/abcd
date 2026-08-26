"""Evaluate one trained model on its held-out views."""

import tyro

from abcd.evaluate import evaluate

if __name__ == "__main__":
    tyro.cli(evaluate)
