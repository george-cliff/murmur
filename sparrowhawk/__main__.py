# stdlib
import subprocess
import sys

# third-party

# local
from tools.log import start_logging, write_log
from tools.paths import SPARROWHAWK_LOG_DIR

SCOPE = "PIPELINE"

# each stage runs as its own process so latest_run_id() in export.py/quantise.py
# re-checks the disk fresh, instead of seeing a stale checkpoint from before train.py ran
STAGES = ["sparrowhawk.train", "sparrowhawk.export", "sparrowhawk.quantise", "sparrowhawk.eval"]


def run_stage(module_name):
    write_log(SCOPE, "started stage", data_module=module_name)
    result = subprocess.run([sys.executable, "-m", module_name])
    if result.returncode != 0:
        write_log(
            SCOPE, "stage failed", data_module=module_name, data_returncode=result.returncode
        )
        sys.exit(result.returncode)
    write_log(SCOPE, "finished stage", data_module=module_name)


if __name__ == "__main__":
    start_logging(SPARROWHAWK_LOG_DIR, "pipeline")
    for stage in STAGES:
        run_stage(stage)
