"""Prepare then run the locked restricted-ownership experiment; no expansion."""

import argparse
from pathlib import Path

from zimortal.training.mortal_actor_center_probe import DEFAULT, prepare, run

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--directory", type=Path, default=DEFAULT)
    args = parser.parse_args()
    (prepare if args.prepare else run)(args.directory)
