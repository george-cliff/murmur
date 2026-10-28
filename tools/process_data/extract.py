# stdlib
from pathlib import Path
import shutil
import zipfile

# third-party

# local
from tools.log import start_logging, write_log
from tools.paths import DATA_DIR, RAW_DIR, TOOLS_LOG_DIR

SCOPE = "EXTRACT"

zips = [
    (f"{DATA_DIR}/warblrb10k_public_wav.zip", f"{RAW_DIR}/train/warblrb10k/"),
    (f"{DATA_DIR}/ff1010bird_wav.zip", f"{RAW_DIR}/train/freefield1010/"),
    (f"{DATA_DIR}/BirdVox-DCASE-20k.zip", f"{RAW_DIR}/train/birdvox/"),
]

csvs = [
    (f"{DATA_DIR}/warblrb10k_public_metadata_2018.csv", f"{RAW_DIR}/train/warblrb10k/"),
    (f"{DATA_DIR}/ff1010bird_metadata_2018.csv", f"{RAW_DIR}/train/freefield1010/"),
    (f"{DATA_DIR}/BirdVoxDCASE20k_csvpublic.csv", f"{RAW_DIR}/train/birdvox/"),
]


def run_extract():
    # each step checks if it's already done first, so reruns don't crash or redo work
    for zip_path, dest_path in zips:
        if Path(dest_path).exists() and any(Path(dest_path).iterdir()):
            write_log(SCOPE, "skipped zip", file_zip=zip_path, file_reason="already extracted")
            continue
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(dest_path)
        write_log(SCOPE, "extracted zip", file_zip=zip_path)

    for csv_path, dest_path in csvs:
        if not Path(csv_path).exists():
            write_log(SCOPE, "skipped csv", file_csv=csv_path, file_reason="already moved")
            continue
        shutil.move(csv_path, dest_path)
        write_log(SCOPE, "moved csv", file_csv=csv_path, file_dest=dest_path)

    for zip_path, _ in zips:
        zip_dest = f"{DATA_DIR}/zips/"
        if not Path(zip_path).exists():
            continue
        shutil.move(zip_path, zip_dest)
        write_log(SCOPE, "moved zip", file_zip=zip_path, file_dest=zip_dest)

    write_log(SCOPE, "completed extraction")


if __name__ == "__main__":
    start_logging(TOOLS_LOG_DIR, "extract")
    run_extract()
