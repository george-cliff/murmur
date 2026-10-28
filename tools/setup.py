# stdlib
import os

# third-party

# local
from tools.log import start_logging, write_log
from tools.paths import (
    DATA_DIR,
    RAW_DIR,
    PROCESSED_DIR,
    SPARROWHAWK_MODEL_DATA_DIR,
    PERCH_MODEL_DATA_DIR,
    TOOLS_LOG_DIR,
    SPARROWHAWK_LOG_DIR,
    MODELS_DIR,
)

SCOPE = "SETUP"

FOLDERS = [
    f"{RAW_DIR}/train/warblrb10k/wav/",
    f"{RAW_DIR}/train/freefield1010/wav/",
    f"{RAW_DIR}/train/birdvox/wav/",
    f"{PROCESSED_DIR}/train/warblrb10k/wav/",
    f"{PROCESSED_DIR}/train/freefield1010/wav/",
    f"{PROCESSED_DIR}/train/birdvox/wav/",
    f"{DATA_DIR}/zips/",
    f"{SPARROWHAWK_MODEL_DATA_DIR}/",
    f"{PERCH_MODEL_DATA_DIR}/",
    f"{SPARROWHAWK_LOG_DIR}/",
    f"{TOOLS_LOG_DIR}/",
    f"{MODELS_DIR}/",
]


def run_setup():
    count_new = 0
    count_exist = 0
    for path in FOLDERS:
        if not os.path.exists(path):
            os.makedirs(path)
            count_new += 1
        else:
            count_exist += 1

    write_log(SCOPE, "created folders", count_created=count_new, count_existed=count_exist)


if __name__ == "__main__":
    start_logging(TOOLS_LOG_DIR, "setup")
    run_setup()
