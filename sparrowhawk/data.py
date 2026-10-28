# stdlib

# third-party
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

# local
from tools.log import write_log
from tools.paths import MANIFEST_PROCESSED_PREFIX, PROCESSED_DIR

# ============================================================
# CONSTANTS
# ============================================================

SCOPE = "DATA"

FOLD_MAP = [
    {"train": ["ff1010bird", "BirdVox-DCASE-20k"], "test": ["warblrb10k"]},
    {"train": ["ff1010bird", "warblrb10k"], "test": ["BirdVox-DCASE-20k"]},
    {"train": ["warblrb10k", "BirdVox-DCASE-20k"], "test": ["ff1010bird"]},
]
FULL_CLIP = True
MAX_MASK_WIDTH = 20
NOISE_WEIGHT = 0.2


# ============================================================
# MANIFEST HELPERS
# ============================================================


def get_manifest():
    manifest = pd.read_csv(f"{PROCESSED_DIR}/manifest.csv")
    manifest["filepath"] = manifest["filepath"].str.replace(
        MANIFEST_PROCESSED_PREFIX, PROCESSED_DIR, n=1, regex=False
    )
    return manifest


def get_fold(i):
    fold = FOLD_MAP[i]
    train = fold["train"]
    test = fold["test"]
    return train, test


def filter_manifest(manifest, datasets):
    ids = manifest["datasetid"]
    mask = ids.isin(datasets)
    filtered_df = manifest[mask]
    return filtered_df


# ============================================================
# AUGMENTATION HELPERS
# ============================================================


def _apply_masking(clip, rng):
    masked_clip = clip.copy()
    # frequency mask
    f_start = rng.integers(masked_clip.shape[0])
    f_width = rng.integers(MAX_MASK_WIDTH)
    masked_clip[f_start : f_start + f_width, :] = 0

    # time mask
    t_start = rng.integers(masked_clip.shape[1])
    t_width = rng.integers(MAX_MASK_WIDTH)
    masked_clip[:, t_start : t_start + t_width] = 0

    return masked_clip, f_start, f_width, t_start, t_width


def _apply_green_noise(clip, rng):
    white_noise = rng.normal(size=clip.shape)
    weights = np.linspace(-1, 1, clip.shape[0])
    bell_curve = np.exp(-(weights**2))
    bell_curve = np.expand_dims(bell_curve, axis=1)
    green_noise = white_noise * bell_curve
    return clip * (1 - NOISE_WEIGHT) + green_noise * NOISE_WEIGHT


def _fit_to_target(spect_array, rng):
    if FULL_CLIP:
        target = 1000
        if spect_array.shape[1] < target:
            clip = np.pad(spect_array, ((0, 0), (0, target - spect_array.shape[1])))
        else:
            clip = spect_array[:, :target]
    else:
        target = 500
        # Important! don't change <= that `=` is load bearing, rng will crash on 0 range if removed
        if spect_array.shape[1] <= target:
            clip = np.pad(spect_array, ((0, 0), (0, target - spect_array.shape[1])))
        else:
            start = rng.integers(spect_array.shape[1] - target)
            clip = spect_array[:, start : start + target]
    return clip


# ============================================================
# DATASET
# ============================================================


class BirdDataset(Dataset):
    def __init__(self, df, fold=0, seed=0, augment_noise=False, augment_mask=False):
        self.df = df
        self.seed = seed
        self.fold = fold
        self.epoch = 0
        self.augment_noise = augment_noise
        self.augment_mask = augment_mask

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # unique random seed per clip, based off seed/idx/fold/epoch for reproducibility
        if not FULL_CLIP:
            clip_seed = self.seed + idx + self.fold + self.epoch * len(self.df)
        else:
            clip_seed = 42
        rng = np.random.default_rng(clip_seed)
        row = self.df.iloc[idx]
        filepath = row["filepath"]
        label = row["hasbird"]
        spect_array = np.load(filepath)
        clip = _fit_to_target(spect_array, rng)
        # log-scale + normalise are baked into the .npy files already during preprocessing, so does not need to be added here
        if self.augment_noise:
            clip = _apply_green_noise(clip, rng)

        if self.augment_mask:
            clip, f_start, f_width, t_start, t_width = _apply_masking(clip, rng)

        return torch.tensor(clip, dtype=torch.float32).unsqueeze(0), label

    def set_epoch(self, epoch):
        self.epoch = epoch


# ============================================================
# DATALOADER HELPERS
# ============================================================


# wraps DataLoader with a single log line - name says which dataset
# (train/val/test), fold/seed are optional since not every dataloader
# (e.g. val/test) is rebuilt per-seed
def build_dataloader(dataset, name, fold=None, seed=None, **dataloader_kwargs):
    dataloader = DataLoader(dataset, **dataloader_kwargs)
    context = {}
    if fold is not None:
        context["fold_number"] = fold
    if seed is not None:
        context["seed_number"] = seed
    write_log(
        SCOPE, "built dataloader", data_name=name, data_rows=len(dataset), **context
    )
    return dataloader


if __name__ == "__main__":
    for i in range(3):
        print(f"---FOLD {i}---")
        manifest = get_manifest()
        train_names, test_names = get_fold(i)
        print(f"train: {train_names}, test: {test_names}")
        train_df = filter_manifest(manifest, train_names)
        test_df = filter_manifest(manifest, test_names)
        total = len(train_df) + len(test_df)
        print(
            f"train rows: {len(train_df)} ({len(train_df) / total:.0%}), test rows: {len(test_df)} ({len(test_df) / total:.0%})"
        )
        print(train_df["hasbird"].value_counts())
        print("--------")
