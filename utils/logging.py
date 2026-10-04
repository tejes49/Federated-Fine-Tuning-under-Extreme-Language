"""Console logger plus a JSON-lines experiment log (real measurements only)."""
import json
import logging
import time
from pathlib import Path


def get_logger(name: str = "fedlang") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(h)
        logger.setLevel(logging.INFO)
    return logger


class ExperimentLog:
    """Append one JSON record per event to results/<run>/log.jsonl."""

    def __init__(self, run_dir: str):
        self.path = Path(run_dir)
        self.path.mkdir(parents=True, exist_ok=True)
        self.file = self.path / "log.jsonl"

    def log(self, **record) -> None:
        record["time"] = time.time()
        with open(self.file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
