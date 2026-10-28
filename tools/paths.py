import os

ON_KAGGLE = os.path.exists("/kaggle/")

if ON_KAGGLE:
    DATA_DIR = "/kaggle/input/datasets/georgecliff/processed-dcase-sparrowhawk"
    LOGS_DIR = "/kaggle/working/logs"
    MODELS_DIR = "/kaggle/working/models"
    KAGGLE_RUN_COUNTER = "sparrowhawk/.run_counter"
    RAW_DIR = f"{DATA_DIR}/raw"
    PROCESSED_DIR = f"{DATA_DIR}/processed"
else:
    DATA_DIR = "data"
    LOGS_DIR = "logs"
    MODELS_DIR = "models"
    KAGGLE_RUN_COUNTER = (
        None  # set to none as experiment.py imports it regardless of platform
    )
    RAW_DIR = f"{DATA_DIR}/sparrowhawk/raw"
    PROCESSED_DIR = f"{DATA_DIR}/sparrowhawk/processed"

# filepath columns in existing manifest CSVs were baked in with this prefix at generation time
# (back when PROCESSED_DIR was "data/processed", before the data/sparrowhawk/ reorganisation).
# get_manifest() rewrites it to wherever PROCESSED_DIR actually resolves to now - locally after
# the move, or to Kaggle's real mount path when running there.
MANIFEST_PROCESSED_PREFIX = "data/processed"

MODEL_DATA_DIR = f"{DATA_DIR}/model_data"
SPARROWHAWK_MODEL_DATA_DIR = f"{MODEL_DATA_DIR}/sparrowhawk"
PERCH_MODEL_DATA_DIR = f"{MODEL_DATA_DIR}/perch"
TOOLS_LOG_DIR = f"{LOGS_DIR}/tools"
SPARROWHAWK_LOG_DIR = f"{LOGS_DIR}/sparrowhawk"
