# stdlib
import os
from pathlib import Path

# third-party

# local
from tools.paths import KAGGLE_RUN_COUNTER


def next_run_id(prefix, log_dir):
    if prefix == "K":
        count = 0
        if os.path.exists(KAGGLE_RUN_COUNTER):
            with open(KAGGLE_RUN_COUNTER) as f:
                count = int(f.read().strip())
        count += 1
        with open(KAGGLE_RUN_COUNTER, "w") as f:
            f.write(str(count))
        return f"{prefix}{count}"
    if not os.path.exists(log_dir):
        return f"{prefix}1"
    existing = [f for f in os.listdir(log_dir) if f.startswith(f"sparrowhawk-{prefix}")]
    return f"{prefix}{len(existing) + 1}"


def latest_run_id(prefix, models_dir):
    matches = list(Path(models_dir).glob(f"sparrowhawk-{prefix}*.pt"))
    if not matches:
        raise FileNotFoundError(f"No {prefix}-prefixed checkpoints found in {models_dir}")
    numbers = [int(f.stem.removeprefix(f"sparrowhawk-{prefix}")) for f in matches]
    return f"{prefix}{max(numbers)}"
