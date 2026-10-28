# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.8.0] - 2026-07-07

### Added

- `Cargo.toml` added dependencies:
  - `rusqlite` (`bundled` feature) added.
- `src/db.rs` created:
  - `init_db()` opens `data/murmur.db`, turns on `PRAGMA foreign_keys`, and runs `schema.sql`.
  - `seed_perch_model_version()`/`get_or_create_model_label()` get-or-create the reference rows a detection needs to link against.
- `src/unit.rs` created:
  - `register_unit()` registers a unit by its caller-supplied id (a ULID, eventually..).
  - `deploy_unit()` get-or-creates the one open deployment for a unit, relying on the schema's own partial unique index (one open deployment per unit).
- `src/detection.rs` rebuilt as the orchestration + persistence layer:
  - `run_models()` seeds a Perch `model_versions` row, runs the full Sparrowhawk → Perch pipeline per WAV file, and writes a `detections` row for every Perch chunk hit via the new `insert_detection()`.
  - Detections are deduplicated via `sequence_no`, so re-running the program doesn't create duplicate rows.
- `src/main.rs`'s `register_and_deploy_unit()` registers the test unit and its deployment.
- `src/perch.rs`'s `identify_species()`/`top_prediction()` now also return the Perch class index (previously computed internally and discarded) — needed for `model_labels.class_index`.

### Changed

- `src/detection.rs` renamed to `src/sparrowhawk.rs`
- `src/species.rs` renamed to `src/perch.rs`
- `schema.sql` - every `CREATE ...` given `IF NOT EXISTS`.

## [0.7.0] - 2026-07-06

### Added

- `Cargo.toml` added dependencies:
  - `reqwest` (`blocking`, `json` features), `serde` (`derive`), `serde_json`, `moontool` dependencies added.
- `src/open_meteo.rs` scaffoled created (dangerous `unwraps` ahead):
  - `fetch_weather()` calls Open-Meteo's forecast API, returns `(current_weather, daily_weather)` JSON trimmed to the fields `schema.sql` needs.
  - `fetch_air_quality()` calls Open-Meteo's air-quality API, returns trimmed `current_air_quality` JSON.
  - `build_url()`/`fetch_json()` shared helpers: URL built via `reqwest::Url`'s `query_pairs_mut()`.
- `src/moon.rs` created:
  - `fetch_moon_phase()` computes the current moon phase via the `moontool` crate, returns `moon_phases` JSON.
- `src/main.rs` updated:
  - `get_enviroment()` calls all three fetches and writes the merged result to `output/environment.json`, keyed by table name.
- `schema.sql` updated:
  - `weather_observations` renamed to `current_weather`, expanded with every field Open-Meteo's current-conditions block returns (apparent temperature, surface pressure, wind gusts/direction, rain/showers/snowfall, cloud cover, weather code, is-day); every measurement column given an explicit unit suffix.
  - `daily_weather` table added (write-once, keyed by lat/lon grid bucket + date), covering Open-Meteo's daily summary block.
  - `current_air_quality` table added (AQI, particulates, pollen counts).
  - `moon_phases` reworked: fields are now `phase`, `illumination_pct`, `age_days`, `distance_km`.
  - `deployments` gained `elevation_m`.
  - `detections` gained `current_weather_id`/`daily_weather_id`/`current_air_quality_id` foreign keys.
- `src/model.rs` created:
  - `load_model` loads an ONNX file into a `Session`.
  - `sigmoid` converts a raw logit to a 0-1 confidence score.
  - `run_inference` runs a session against a given input/output tensor name and returns the raw output values.
- `src/species.rs` created:
  - `load_labels` reads the species name list from `labels.csv`.
  - `get_chunk` extracts a fixed-size window of samples starting at a given position, zero-padding if not enough audio remains.
  - `to_input_array` reshapes a chunk into Perch's 2D input shape.
  - `to_tensor` wraps a reshaped array as an `ort::Tensor`.
  - `top_prediction` finds the highest-scoring species and its confidence from Perch.
  - `identify_species` splits a WAV's audio into two 5-second chunks and returns a species/confidence prediction for each.
- `src/main.rs` updated:
  - Loads the Perch model and species labels once, then runs species identification on any WAV Sparrowhawk flags as `bird`, printing both chunks' predictions.

### Removed

- `schema.sql` - `solar_times` table removed; `sunrise_time`/`sunset_time` folded into `daily_weather` instead, since Open-Meteo's daily block already provides them and the separate table was duplicating that data. `detections.solar_time_id` removed.

### Changed

- `src/detection.rs`:
  - `load_model`, `sigmoid`, and `run_inference` moved to `model.rs`, since Perch needed the exact same logic.
  - `run_inference` now takes `input_name`/`output_name` parameters, so the shared version works for both Sparrowhawk (`"mel_spectrogram"`/`"raw_logit"`) and Perch (`"inputs"`/`"label"`).
- `src/audio.rs` split `audio_pipeline`:
  - `audio_pipeline` now only loads and resamples a WAV to 32kHz, returning raw samples.
  - `spect_pipeline` takes samples directly and does the STFT/mel/log work, so Perch (raw waveform) and Sparrowhawk (spectrogram) can both build on the same shared loading step.
- `src/main.rs`, `src/detection.rs` updated to call `audio_pipeline` then `spect_pipeline` as two explicit steps.

## [0.6.0] - 2026-07-05

### Added

- `Cargo.toml` added dependencies:
  - `ort`, `rubato` dependencies added.
- `src/detection.rs` updated:
  - `load_model` loads an ONNX file into a `Session`.
  - `load_threshold` reads the deployment threshold from `sparrowhawk-V9_threshold.txt` at runtime.
  - `fit_to_target` crops or zero-pads a spectrogram to a fixed column count.
  - `to_input_array` reshapes a spectrogram to the model's 4D input shape.
  - `to_tensor` wraps a reshaped array as an `ort::Tensor`.
  - `run_inference` runs the model and extracts the raw logit.
  - `sigmoid` converts a raw logit to a 0-1 confidence score.
  - `detect` ties the above together into a single `(is_bird, confidence)` result for one WAV file.
- `src/audio.rs` updated:
  - `resample_to_32k` resamples a WAV's samples to 32kHz via `rubato`'s `Fft` resampler, matching the training data resampling.
- `src/main.rs` updated:
  - Loads the model and threshold once, then runs inference on six WAV files (a `hasbird = 1` and a `hasbird = 0` from each dataset) and prints `bird`/`not bird` with a confidence score for each.

### Fixed

- `src/audio.rs` fixes:
  - `audio_pipeline` now returns the raw log-scaled spectrogram; the min-max rescale (`normalise_mel`) is only applied in `main.rs`'s PNG debug output path.
  - resamples every WAV to 32kHz immediately after loading, matching the sample rate sparrowhawk was trained on. was previously just orignal sample rate (44kHz).
- `src/main.rs` fixes:
  - Test WAV file paths now point at `data/sparrowhawk/raw/...` rather then previous `data/raw/...`.

## [0.5.0] - 2026-07-04

### Added

- `sparrowhawk/quantise.py` - calibrates on manifest rows outside the val split, runs static INT8 quantisation (QDQ format) on the exported ONNX model, and logs float32-vs-quantised AUC/accuracy/average-inference-time on val set.
- `sparrowhawk/eval.py` - sweeps thresholds (via `BETA_MAP`/`THRESHOLD_MAP`) against the quantised model's val-set scores and saves the chosen deployment threshold to `data/model_data/sparrowhawk/sparrowhawk-V{n}_threshold.txt`.
- `sparrowhawk/utils.py` - `next_run_id()`/`latest_run_id()` moved here out of `experiment.py`/`train.py` so every stage (train, export, quantise, eval) can find the right run without importing each other.
- `sparrowhawk/__main__.py` - runs train, export, quantise, and eval as four separate subprocesses in sequence, so each stage sees the previous stage's output on disk instead of a stale in-memory state.
- `train.py` now saves its exact validation split to `data/model_data/sparrowhawk/sparrowhawk-V{n}_val.csv` alongside the checkpoint, so quantisation/eval calibrate and score against the same held-out rows the model was validated on.
- Per-100-batch progress logging in `run_epoch` (`data.batch`/`data.batch_total`/`metric.running_loss`), so a long epoch doesn't look stuck.
- GPU OOM workaround: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` set in `train.py` before `torch` is imported.

### Changed

- **Architecture:** conv block 2 split into `conv2a`/`conv2b` (same depth-split pattern already used on block 1), `CONV4_OUT` reverted from 256 back to 128. New baseline: `BASELINE_AUC`/`BASELINE_UNW_LOSS`/`BASELINE_RECALL` updated from 0.8355/0.7599/0.6267 to 0.8368/0.7132/0.6062.
- `train_final.py` renamed to `train.py` (as it is now the main single-run training pipeline); the old `train.py` (hyperparameter/architecture tuning loop) renamed to `experiment.py`. Every cross-import between the two updated to match.
- `export.py` now auto-targets the newest `V`-numbered checkpoint via `latest_run_id()` instead of a hardcoded `sample_model` path.
- `data/` reorganised: `data/models/` -> `data/model_data/{perch,sparrowhawk}/`, and `data/raw`+`data/processed` -> `data/sparrowhawk/` (training data specific to this model). `tools/paths.py`, `tools/setup.py`, and `sparrowhawk/data.py` updated to match, including a `get_manifest()` hardcoded path literal (`"data/processed"`) replaced with a named constant (`MANIFEST_PROCESSED_PREFIX`) so old manifests still resolve correctly after the move.

### Removed

- `sparrowhawk/bird.py`, `sparrowhawk/eval.py` (old), and `sparrowhawk/combine.py` - written by Gemini in a draft session without review.

### Fixed

- Fixed `train_final.py` crash by replacing `torch.manual_seed(SEED)` with a valid integer seed in `train_test_split`.
- Applied a 32-bit modulo mask (`SEED % 4294967296`) to ensure the training seed falls within `scikit-learn`'s required unsigned integer range `[0, 4294967295]`.
- Fixed the train/eval leakage bug at its root: `train.py`'s val split is now saved to disk and reused by `quantise.py`/`eval.py`, instead of each script re-deriving its own split.

## [0.4.1] - 2026-06-23

### Fixed

- `README.md` - removed a stale "NO WAY OF SAVING MODELS ATM" note left over from before `train_final.py`/model saving existed.
- `CHANGELOG.md` - added the missing `[0.4.0]` heading and version links after realising `v0.4.0` was tagged without them.

## [0.4.0] - 2026-06-23

### Added

- `docs/logging-standards.md` - standised project-wide logging conventions to be impleneted going forward
- `tools/log.py`: `write_log()` - single helper that builds and prints every log line in the new `[timestamp - SCOPE] message :: field=value | field=value` format, replacing hand-rolled `print(f"[{ts()}] [{LABEL}] ...")` calls.

### Changed

- Every `tools/` and `sparrowhawk/` script's logging converted to `write_log()` and the new format - static lowercase messages, real fields instead of interpolated strings, present-tense "in progress" lines dropped in favour of completion-only events.
- `tools/log.py`'s `ts()` now returns ISO 8601 UTC timestamps (`YYYY-MM-DDTHH:MM:SSZ`) instead of local `HH:MM:SS`.

### Removed

- Eval dataset handling (`warblrb10k_test`/`chernobyl`/`poland_nfc`) removed from `tools/setup.py`, `tools/process_data/extract.py`, and `tools/process_data/preprocess.py` - no ground-truth labels were ever released for these sets (see `sparrowhawk/evaluate.py`'s removal, `0.3.0`), so downloading/extracting/preprocessing them served no purpose. `preprocess.py`'s `process_files()` also dropped its now-always-`"train"` `label` parameter.

## [0.3.0] - 2026-06-23

### Added

- Kaggle support:
  - `tools/paths.py` auto-detects Kaggle (`ON_KAGGLE`) and switches data/log/model paths to Kaggle's `/kaggle/input/`/`/kaggle/working/` layout, so the same code runs on desktop or Kaggle unmodified.
  - `sparrowhawk/.run_counter` a git-tracked counter for Kaggle's `RUN_ID` numbering, since `/kaggle/working/` resets every run and can't be counted from directly.
  - `notebooks/sparrowhawk-train.ipynb` created for kaggle runs
- `sparrowhawk/train_final.py` - scaffolded the eventual final training run: trains on the *whole* manifest with no fold held out. (currently fully untested)

### Changed

- New baseline: `sparrowhawk/train.py`'s `BASELINE_AUC`/`BASELINE_UNW_LOSS`/`BASELINE_RECALL` updated from `0.7930`/`1.0115`/`0.584` to `0.8355`/`0.7599`/`0.6267` after a batch of capacity experiments (channel widening, depth, kernel size). Promoted run `L4`: depth split on conv block 1, conv2 widened to 64, 5×5 kernel.
  - `sparrowhawk/model.py`: first conv stage split into `conv1a`→`conv1b` (each with its own batch norm), taking inspiration from a DCASE model, `Vggish`'s `conv→conv→pool` block pattern instead of a single conv per stage.
  - Added a `PADDING` constant derived from `CONV_KERNEL`, applied to every conv layer - stops the spectrogram's edges (low/high frequency bands, clip start/end) getting cropped away layer after layer.
- Replaced `train.py`'s hand-set `BASELINE`/`VERSION` constants with a single auto-incrementing `RUN_ID` (`K1`, `L1`, depending on Kaggle or local), counted from existing logs. Log filenames now `sparrowhawk-{RUN_ID}-T{timestamp}.log`.
- `NUM_WORKERS` now platform-dependent (`4` on Kaggle, `6` local) - Kaggle's GPU instance only has 4 CPU cores available.
- `sparrowhawk/sparrowhawk.py` renamed to `sparrowhawk/__main__.py` - parked for a future single entry point (`python -m sparrowhawk`) chaining the whole pipeline (train, save, export). `sparrowhawk.py` didn't actually do this - Python only looks for `__main__.py` specifically, not a file matching the package's own name.
- `train.py`'s `_run_epoch`, `_format_model_layers`, `_sweep_thresholds`, and `_next_run_id` renamed to drop their leading underscore (`run_epoch`, `format_model_layers`, `sweep_thresholds`, `next_run_id`) - the underscore convention meant "nothing outside this file imports these," which stopped being true once `train_final.py` started reusing them.

### Removed

- `sparrowhawk/evaluate.py` (empty stub) - planned eval script dropped after being unable to find the intended eval sets (`chernobyl`/`poland_nfc`/`warblrb10k_test`) labels from DCASE 2018.

### Fixed

- Fixed local training OOMs on 16GB system RAM (kernel killing desktop processes to free memory, which ended up killing training loops on my poor PC): `test_dataloader` no longer uses `persistent_workers=True`, and `NUM_WORKERS` dropped 8→6.
- Fixed manifest having local-only paths baked into its `filepath` column.
- Moved `signal.signal(...)` (the Ctrl+C handler) into `train()` in `train.py`, since registering it at module level meant just *importing* the file (e.g. from `train_final.py`) silently registered it too.

## [0.2.0] - 2026-06-19

### Added

- `requirements.txt` added dependencies:
  - `numpy`, `scikit-learn`, `librosa` dependencies added.
- `sparrowhawk/data.py` created:
  - `BirdDataset`: Hands one spectrogram and its bird/not-bird label to the model at a time during training. Uses the full 10-second clip by default (`FULL_CLIP`), or can pick a 5-second window instead - the same setup always picks the same window, so results are reproducible.
  - 3-way cross-validation (`FOLD_MAP`) - each fold trains on 2 of the 3 DCASE 2018 datasets and tests on the dataset left out, as recommended by DCASE community.
  - Reads and loads pre-computed `.npy` spectrograms.
  - Training augmentation - only ever applied to training data (`augment_noise`/`augment_mask` flags, one per technique so they can be tested independently or combined), validation/test clips are always left untouched:
    - `apply_masking()`: masks a random band of frequencies and a random slice of time on every training spectrogram, forcing the model to learn from incomplete/noisy input instead of memorising clean clips.
    - `apply_green_noise()`: generates synthetic green noise (white noise weighted by a bell curve centred on the middle mel-frequency bin, so energy is concentrated mid-band rather than flat) and adds it on top of the real clip at a fixed weight (`NOISE_WEIGHT`), dropping the signal-to-noise ratio without needing any real noise recordings as a source.
- `sparrowhawk/train.py` created:
  - Full training loop implemented:
    - Each fold's training data is further split into training, validation, and test sets.
    - Automatically lowers the learning rate when validation performance stalls using `ReduceLROnPlateau`.
    - Stops training automatically once it stops improving, and reverts to the highest scoring model for testing.
    - Uses an F-beta sweep (via `BETA_MAP`) after training to pick the best decision threshold.
  - Trains across all 3 DCASE folds.
  - Created `SMOKE_TEST`/`SMOKE_TEST_FOLDS`/`SMOKE_TEST_SEEDS` parameters to control run size before committing to a full experiment:
    - Due to limitations of my hardware (my trusty ol' reliable *RTX 2060 6GB*), running every queued idea at full size would take days. I'm following a two-tiered smoke test system for experiments:
      - tier 1 is a quick check (3 folds x 3 seeds, 9 runs) every idea gets first.
      - tier 2 is a longer confirmation run for ideas that look promising, anywhere from 3 folds x 5 seeds (15 runs) if it's an expected result, to 3 folds x 8 seeds (24 runs) if the result was surprising or uncertain and needs more evidence.
  - `cudnn.deterministic`/`benchmark` flags for reproducible reruns.
  - `harmonic_auc`: the harmonic mean of each fold's mean AUC, following DCASE's own scoring method - this punishes a fold that doesn't generalise well, instead of letting two strong folds average it away and hide the weakness. Printed after every run as an informational "would this beat the current baseline?" check - promoting it to the real baseline is still a manual decision.
  - `BASELINE_UNW_LOSS`: tracked the same way as `BASELINE_AUC`, with its own "Beat current baseline?" print - safe to compare across runs that change `pos_weight`, since `unweighted_test_loss` always uses `pos_weight=1` regardless of what's used for training.
  - Startup log line confirming which augmentations are active (`AUGMENT_NOISE`/`AUGMENT_MASK`) and their associated constants.
  - Logs to `logs/sparrowhawk/` via `start_logging()` helper (`tools/log.py`).
  - Early exit handling via *Ctrl+C* - finishes the current epoch, reverts to best checkpoint.
    - AUC/recall/accuracy/false positive/negative counts tracked per-seed.
  - Per-epoch data-loading vs. GPU-compute time split logged.
  - Calculates accuracy, recall, and F-beta scores, missed birds, and false alarms - using `scikit-learn`'s functions.
  - Startup log prints architecture as plain layer/type/shape.
- `tools/` added as a Python package (`__init__.py`):
  - `log.py` - timestamps every log line and sends everything printed to the screen into a log file too, via `start_logging(log_dir, name)`.
    - Bracketed `[HH:MM:SS]` timestamp on every log line.
    - Bracketed `[SCOPE]` tag on every log line.
  - `setup.py` - creates every gitignored working folder the pipeline needs (`data/`, `logs/`, `models/`) - standalone, not specific to the data-prep pipeline.
  - `paths.py` - created for single source of constants of paths that python code needs (`DATA_DIR`, `RAW_DIR`, `PROCESSED_DIR`, `TOOLS_LOG_DIR`, `SPARROWHAWK_LOG_DIR`, `MODELS_DIR`)
  - `process_data/` package created:
    - Turns raw audio recordings into the numeric format the model trains on (a spectrogram, with a logarithmic scaling step applied, saved to a `.npy` file), and writes a manifest file listing every clip and its label.
  - `benchmark_workers.py` - times how fast training data loads with different numbers of background loading processes, so you can pick the fastest setting instead of guessing (tested against the real training dataset, masking included):
    - Script fully generated by Claude Code.
  - Every script above logs to `logs/tools/` via `start_logging()`, named after itself (`extract-T...`, `preprocess-T...`, etc.).
  - Run any script via `python -m tools.<module>` (or `python -m tools.process_data.<module>`) rather than as a direct script - needed because the scripts import each other internally.
- `logs/sparrowhawk/` added - timestamped `.log` file per training run (gitignored).
- `logs/tools/` added - timestamped `.log` file per tool run (gitignored).
- `models/` added - output folder for trained model files (gitignored).
- `docs/model-tuning.md` added - table tracking every queued training experiment for Sparrowhawk (what's expected, what's currently running, completed, and whether it's been applied to the baseline) as well as descriptons of how the tests have gone.

### Changed

- `sparrowhawk/` converted to a Python package, so the files inside it can cleanly reuse each other's code:
  - `__init__.py` added.
  - Internal imports changed to relative (`from .data import ...`, `from .model import ...`).
  - Now run via `python -m sparrowhawk.train`.
- `sparrowhawk/model.py` model architecture updated:
  - Architecture widened from 3 conv blocks (16→32→64) to 4 conv blocks (16→32→64→128).
  - Batch norm added after every conv layer.
  - Plain ReLU replaced with Leaky ReLU throughout.
  - GAP widened from `(1,1)` to `(2,2)`, `fc1` input size updated to match.
  - `sigmoid` removed from the end of `forward()` - loss function now expects raw logits.
  - Some hardcoded architecture numbers moved out into named constants (`CONV1_OUT`-`CONV4_OUT`, `GAP_H`/`GAP_W`, `CONV_KERNEL`, `NEG_SLOPE`).
  - Layers in `__init__` reordered to group by type (all convs, then fc layers, then batch norms, then GAP/flatten/dropout) instead of one interleaved block, matching the layout of the DCASE reference baseline.
- `README.md`:
  - Project banner and one-line description added.
  - Names table restructured with Why/What is it/What does it do columns.
  - **Roost** renamed to **Nest** (the home base server); **Roost** is now the name of the dashboard that visualises detections, a separate component from the server.
  - "how to run" section added.
- `scripts/` replaced by `tools/process_data/`.
- `Cargo.toml` version bumped to `0.2.0` to match the tagged release.  (was 0.0.1 before, whoops...)
- `.gitignore` excludes `logs/`, `models/`, `__pycache__/`, and `*.pyc`.
- `requirements.txt` - CUDA install comment moved to sit directly above `torch` (previously also sat above `torchvision`, which is now removed).
- `CHANGELOG.md` - existing entries rewritten for clearer formatting and better human readability: shorter top-level bullets with nested specifics, consistent punctuation, and reduced jargon.
- `src/main.rs` - replaced the single hardcoded test WAV/output path with a `WAV_MAP` loop, generating one spectrogram PNG per DCASE dataset (named after the dataset folder) instead of just one fixed clip - used to visually compare call clarity across datasets.

### Removed

- `requirements.txt` - `torchvision` dependency removed.

### Fixed

- `tools/process_data/extract.py` - rerunning extraction after a successful run used to crash, because only the zip-extraction step checked whether it had already run. Moving the CSVs and zips would then fail since they'd already been moved. Every step now checks first, so the whole thing can be safely rerun.

## [0.1.2] - 2026-06-14

### Changed

- `audio_pipeline()` now uses the same spectrogram settings as Google's Perch model (`WIN_SIZE` 640, `HOP_SIZE` 320, `FRAME_SIZE` 1024, `F_MIN`-`F_MAX` 60–16,000 Hz), so Model 1's preprocessing stays consistent with Perch's.
- Added a logarithmic scaling step after generating the spectrogram, compressing the gap between quiet and loud sounds so both ends stay usefully visible to the model - matches how Perch preprocesses its own input.
- Renamed functions and variables throughout `audio.rs` and `main.rs` for consistency.

## [0.1.1] - 2026-06-14

### Added

- `spectrs` crate dependency.
- `scripts/setup.py` - creates all required `data/` and `output/` folders.
- `scripts/extract.py` - unzips DCASE 2018 datasets and moves label CSVs into place.
- `scripts/manifest.py` - merges all three development dataset CSVs into a single `data/manifest.csv` (35,690 rows) with filepath, label, and dataset columns.
- `requirements.txt` - Python dependencies for the scripts pipeline.

### Removed

- `rustfft` crate dependency.

### Changed

- Replaced my spectrogram functions with `spectrs` crate, for better results.
- `audio_pipeline()` now returns `Array2<f32>` normalised to 0.0–1.0 for model inference rather than 0.0-255.0 for PNG creation.

## [0.1.0] - 2026-06-13

### Added

- Built the full pipeline that turns a WAV audio file into a spectrogram image:
  - Loads the audio.
  - Slices it into overlapping frames.
  - Extracts frequency content (FFT).
  - Compresses it to the mel scale.
  - Normalises the values.
  - Optionally saves it as a PNG for manual visual testing during development.
- `audio_pipeline()` - end-to-end function from WAV file to normalised mel spectrogram.
- `generate_spectrogram_png()` - saves spectrogram as greyscale PNG for visual inspection.

## [0.0.1] - 2026-06-12

### Added

- Initial project scaffold - `Cargo.toml` and `src/main.rs` with starter content, `src/audio.rs` and `src/detection.rs` as empty placeholder files.
- `README.md` with project overview and names glossary.
- `CHANGELOG.md` and `ROADMAP.md`.

[0.8.0]: https://github.com/george-cliff/murmur/compare/v0.7.0...v0.8.0
[0.7.0]: https://github.com/george-cliff/murmur/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/george-cliff/murmur/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/george-cliff/murmur/compare/v0.4.1...v0.5.0
[0.4.1]: https://github.com/george-cliff/murmur/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/george-cliff/murmur/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/george-cliff/murmur/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/george-cliff/murmur/compare/v0.1.2...v0.2.0
[0.1.2]: https://github.com/george-cliff/murmur/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/george-cliff/murmur/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/george-cliff/murmur/compare/v0.0.1...v0.1.0
[0.0.1]: https://github.com/george-cliff/murmur/releases/tag/v0.0.1
