"""
IMPORTANT BEFORE RUNNING PLEASE:

1. Run tools.setup first to create the required folders

2. Go to https://dcase.community/challenge2018/task-bird-audio-detection
    DOWNLOAD:
        - data labels AND audio files for all 3 development datasets (freefield1010, warblrb10k, and BirdVoc-DCASE-20k)
        - audio files for all 3 evaulation datasets (warblrb10k, Chernobyl, PolandNFC)

    and place them in "murmur/data/"
"""

# stdlib

# third-party

# local
from tools.log import start_logging
from tools.paths import DATA_DIR, TOOLS_LOG_DIR
from tools.process_data.extract import run_extract
from tools.process_data.manifest import write_raw_manifest, write_processed_manifest
from tools.process_data.preprocess import run_preprocess


if __name__ == "__main__":
    start_logging(TOOLS_LOG_DIR, "process_data")
    run_extract()
    write_raw_manifest()  # `.wav` manifest for preprocessing
    run_preprocess()
    write_processed_manifest()  # `.npy` manifest for model
    # final human-readable summary, not a log event — no timestamp/scope tag
    print("==================================================")
    print("  data fully processed — WAV files converted to")
    print("  numpy arrays and ready for training Sparrowhawk")
    print("--------------------------------------------------")
    print(f"  NOTE: zips moved to {DATA_DIR}/zips/ — safe to delete")
    print("  if you need the space")
    print("==================================================")
