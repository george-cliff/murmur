# stdlib

# third-party
import pandas as pd

# local
from tools.log import start_logging, write_log
from tools.paths import RAW_DIR, PROCESSED_DIR, TOOLS_LOG_DIR

SCOPE = "MANIFEST"

# the 3 DCASE 2018 development datasets (train data)
# find them here: https://dcase.community/challenge2018/task-bird-audio-detection
# eval datasets have no manifest since they ship with no labels (held
# back for blind scoring)
datasets = [
    (
        f"{RAW_DIR}/train/warblrb10k/warblrb10k_public_metadata_2018.csv",
        f"{RAW_DIR}/train/warblrb10k/wav/",
    ),
    (
        f"{RAW_DIR}/train/freefield1010/ff1010bird_metadata_2018.csv",
        f"{RAW_DIR}/train/freefield1010/wav/",
    ),
    (
        f"{RAW_DIR}/train/birdvox/BirdVoxDCASE20k_csvpublic.csv",
        f"{RAW_DIR}/train/birdvox/wav/",
    ),
]


def write_raw_manifest():
    df_list = []
    for csv_path, wav_dir in datasets:
        df = pd.read_csv(csv_path)
        df["filepath"] = (
            wav_dir + df["itemid"].astype(str) + ".wav"
        )  # astype(str) due to some files being pure number names
        df_list.append(df)
    manifest = pd.concat(df_list)
    manifest.to_csv(f"{RAW_DIR}/manifest.csv", index=False)
    write_log(
        SCOPE, "wrote raw manifest", data_rows=len(manifest), file_path=f"{RAW_DIR}/manifest.csv"
    )


def write_processed_manifest():
    df_list = []
    for csv_path, wav_dir in datasets:
        df = pd.read_csv(csv_path)
        processed_dir = wav_dir.replace(RAW_DIR, PROCESSED_DIR)
        df["filepath"] = (
            processed_dir + df["itemid"].astype(str) + ".npy"
        )  # astype(str) due to some files being pure number names
        df_list.append(df)
    manifest = pd.concat(df_list)
    manifest.to_csv(f"{PROCESSED_DIR}/manifest.csv", index=False)
    write_log(
        SCOPE,
        "wrote processed manifest",
        data_rows=len(manifest),
        file_path=f"{PROCESSED_DIR}/manifest.csv",
    )


if __name__ == "__main__":
    import sys

    start_logging(TOOLS_LOG_DIR, f"manifest-{sys.argv[1]}")
    if sys.argv[1] == "raw":
        write_raw_manifest()
    elif sys.argv[1] == "proc":
        write_processed_manifest()
