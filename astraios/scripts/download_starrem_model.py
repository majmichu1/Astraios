"""Built-in star removal model stub — no download needed.

Star removal now uses a morphological approach that works on any image
without any external model. This script is kept for API compatibility.
"""

from __future__ import annotations

import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).resolve().parent.parent / "ai" / "inference" / "models"


def download_model(output_path: str | None = None):
    log.info(
        "No download needed — built-in star removal uses morphological algorithm"
    )
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).touch()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Star removal model stub")
    parser.add_argument("--output", "-o", help="Output path", default=None)
    args = parser.parse_args()
    download_model(args.output)
