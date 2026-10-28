# stdlib
from pathlib import Path

# third-party
import librosa
import numpy as np
import pandas as pd

# local
from tools.log import start_logging, write_log
from tools.paths import RAW_DIR, PROCESSED_DIR, TOOLS_LOG_DIR

SCOPE = "PREPROCESS"
MANIFEST = f"{RAW_DIR}/manifest.csv"
OUT_DIR = Path(PROCESSED_DIR)


def spec_path(filepath):
    p = Path(filepath)
    return OUT_DIR / p.parent.relative_to(RAW_DIR) / p.with_suffix(".npy").name


def compute_spec(filepath):
    audio, _ = librosa.load(filepath, sr=32000)
    # these numbers currently match Google's Perch model's own spectrogram params
    # there are likely future plans to expriment with them for Sparrowhawk
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=32000,
        n_fft=1024,
        win_length=640,
        hop_length=320,
        n_mels=128,
        fmin=60,
        fmax=16000,
        center=False,
    )

    mel = (
        np.log(np.maximum(mel, 1e-5)) * 0.1
    )  # log scaling and normalisation done in preprocessing to reduce time spent training Sparrowhawk
    return mel


def process_files(filepaths):
    total = len(filepaths)
    skipped = 0
    for i, filepath in enumerate(filepaths):
        out = spec_path(filepath)
        if out.exists():
            skipped += 1
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        spec = compute_spec(filepath)
        np.save(out, spec)
        if (i + 1) % max(
            1, total // 10
        ) == 0:  # logs every 10% of files to not overwhelm
            computed = i + 1 - skipped
            pct = (i + 1) / total * 100
            write_log(
                SCOPE,
                "logged progress",
                count_percent=f"{pct:.0f}",
                count_computed=computed,
                count_skipped=skipped,
            )
    return total, skipped


def run_preprocess():
    manifest = pd.read_csv(MANIFEST)
    train_files = manifest["filepath"].tolist()

    total, skipped = process_files(train_files)

    write_log(SCOPE, "computed spectrograms", count_computed=total - skipped, count_existed=skipped)


if __name__ == "__main__":
    start_logging(TOOLS_LOG_DIR, "preprocess")
    run_preprocess()
