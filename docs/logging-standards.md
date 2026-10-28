# Murmur Logging Guide

reference used: <https://oneuptime.com/blog/post/2026-01-30-log-message-conventions/view>

## 1. Message format standards

1. Prefix: timestamp + scope tag formated: [*ISO 8601 UTC timestamp* - *SCOPE*] . timestamp stays the same through the year (thanks daylight savings...) and scope is where the log originates from (e.g. AUDIO or MODEL)
2. The message itself: Action + Subject + Outcome, static sentence, all lowercase. (e.g. saved spectogram succesfully)
3. seporator between message and context "::"
4. Context: extra fields tacked on after (path, shape, whatever varies) with field name and result. each field seperated by a "|" . (e.g. path= output/SPECT-112.png)

Full line: [timestamp - SCOPE] message :: field=value | field=value
Full Example Line: "[2026-06-23T14:02:01Z - AUDIO] saved spectogram succesfully :: path=output/SPECT-112.png | size=156.2KiB"

## 2. Attribute naming dictionary

Dot notation namespacing (`namespace.field`), grouped by concept.

### Identity (which run/fold/seed/epoch produced this line)

| Name | Type | Description | Example | Used by |
| --- | --- | --- | --- | --- |
| `fold.number` | int | fold number (1-indexed) | `1` | `train.py` |
| `fold.total` | int | total folds in this run | `3` | `train.py` |
| `seed.number` | int | seed's position within a fold (1-indexed) | `2` | `train.py` |
| `seed.total` | int | total seeds run per fold | `3` | `train.py` |
| `seed.value` | int | actual seed value passed to `torch.manual_seed` | `213243223153243223` | `train.py` |
| `epoch.number` | int | epoch number within a seed | `4` | `train.py`, `train_final.py` |

### Data

| Name | Type | Description | Example | Used by |
| --- | --- | --- | --- | --- |
| `data.name` | string | which dataset a row/dataloader count is for (`"train"`/`"val"`/`"test"`) | `"train"` | `data.py`, `train.py`, `train_final.py`, `benchmark_workers.py` |
| `data.rows` | int | row count for the dataset named in `data.name` on the same line | `27690` | `data.py`, `manifest.py`, `train.py`, `train_final.py`, `benchmark_workers.py` |
| `data.batch_size` | int | dataloader batch size | `48` | `train.py`, `benchmark_workers.py` |
| `data.num_workers` | int | dataloader worker count | `6` | `benchmark_workers.py` |

### Model config

| Name | Type | Description | Example | Used by |
| --- | --- | --- | --- | --- |
| `model.learning_rate` | float | optimiser LR | `0.001` | `train.py` |
| `model.weight_decay` | float | optimiser weight decay | `0.0001` | `train.py` |
| `model.patience` | int | early-stop patience (epochs) | `10` | `train.py` |
| `model.sched_patience` | int | LR scheduler patience (epochs) | `5` | `train.py` |
| `model.lr_factor` | float | LR scheduler drop factor | `0.5` | `train.py` |
| `model.leaky_relu_slope` | float | negative slope in LeakyReLU | `0.01` | `train.py` |
| `model.noise` | bool | green-noise augmentation on/off | `false` | `train.py` |
| `model.noise_weight` | float | green-noise weight | `0.2` | `train.py` |
| `model.masking` | bool | spectrogram masking augmentation on/off | `true` | `train.py` |
| `model.mask_width` | int | max mask width | `20` | `train.py` |
| `model.full_clip` | bool | full 10s clips vs random 5s crops | `true` | `train.py` |
| `model.pos_weight` | float | loss class-imbalance weight, final | `1.307` | `train.py`, `train_final.py` |
| `model.base_pos_weight` | float | raw class-imbalance ratio before multiplier | `1.307` | `train.py` |
| `model.pos_weight_multi` | float | multiplier applied to `pos_weight` | `1` | `train.py` |
| `model.cudnn_deterministic` | bool | cuDNN determinism flag | `true` | `train.py` |
| `model.cudnn_benchmark` | bool | cuDNN autotune flag | `false` | `train.py` |
| `model.torch_version` | string | PyTorch version | `"2.3.0"` | `train.py`, `train_final.py` |
| `model.device` | string | torch device | `"cuda"` | `train.py`, `train_final.py` |
| `model.gpu` | string | GPU name, if CUDA | `"RTX 2060"` | `train.py`, `train_final.py` |
| `model.smoke_test` | bool | whether this run is a smoke test | `true` | `train.py` |

### Metrics

| Name | Type | Description | Example | Used by |
| --- | --- | --- | --- | --- |
| `metric.name` | string | metric name in a summary row | `"AUC"` | `train.py` |
| `metric.mean` | float | mean value across seeds | `0.835` | `train.py` |
| `metric.std` | float | std deviation across seeds | `0.012` | `train.py` |
| `metric.auc` | float | AUC score | `0.835` | `train.py` |
| `metric.accuracy` | float | accuracy score | `0.91` | `train.py` |
| `metric.recall` | float | recall score | `0.62` | `train.py` |
| `metric.train_loss` | float | training loss this epoch | `0.452` | `train.py`, `train_final.py` |
| `metric.val_loss` | float | validation loss this epoch | `0.501` | `train.py`, `train_final.py` |
| `metric.status` | string | `"best"` or `"stale N/PATIENCE"` | `"best"` | `train.py`, `train_final.py` |
| `metric.best_auc` | float | best AUC seen so far this seed | `0.835` | `train.py`, `train_final.py` |
| `metric.false_positives` | int | false positive count | `12` | `train.py` |
| `metric.false_negatives` | int | false negative count | `8` | `train.py` |
| `metric.total_false` | int | false positives + false negatives | `20` | `train.py` |
| `metric.beta` | float | F-beta value being evaluated | `1.0` | `train.py`, `train_final.py` |
| `metric.betas` | string | list of F-betas tested | `"0.5, 1.0, 1.5, 2.0"` | `train.py` |
| `metric.f_beta` | float | F-beta score achieved | `0.812` | `train.py` |
| `metric.threshold` | float | decision threshold | `0.45` | `train.py`, `train_final.py` |
| `metric.lr_before` | float | LR before a scheduler drop | `0.001` | `train.py`, `train_final.py` |
| `metric.lr_after` | float | LR after a scheduler drop | `0.0005` | `train.py`, `train_final.py` |
| `metric.value` | float | generic metric value in eval rows | `0.835` | `train.py` |
| `metric.unweighted_loss` | float | test loss with `pos_weight=1` | `0.76` | `train.py` |

### Baseline

| Name | Type | Description | Example | Used by |
| --- | --- | --- | --- | --- |
| `baseline.metric` | string | which metric this comparison is for | `"auc"` | `train.py` |
| `baseline.value` | float | baseline target for the metric named in `baseline.metric` on the same line | `0.8355` | `train.py` |
| `baseline.beat` | bool | whether this run beat the baseline | `true` | `train.py` |

### Timing

| Name | Type | Description | Example | Used by |
| --- | --- | --- | --- | --- |
| `time.minutes` | float | elapsed time, minutes - the message text (`"finished run"`/`"finished fold"`/`"finished seed"`/`"benchmarked worker count"`) plus any `fold.number`/`seed.number` says which level it measures | `47.1` | `train.py`, `train_final.py`, `benchmark_workers.py` |
| `time.seconds` | float | elapsed time, seconds - same disambiguation as `time.minutes` | `2826` | `train.py`, `train_final.py`, `benchmark_workers.py` |
| `time.data_seconds` | float | per-epoch data-loading time | `0.3` | `train.py`, `train_final.py` |
| `time.compute_seconds` | float | per-epoch GPU compute time | `64.2` | `train.py`, `train_final.py` |

### Files

| Name | Type | Description | Example | Used by |
| --- | --- | --- | --- | --- |
| `file.path` | string | the file path involved | `"data/processed/manifest.csv"` | `manifest.py`, `train_final.py`, `export.py` |
| `file.max_diff` | float | max numeric diff, PyTorch vs ONNX output | `0.000012` | `export.py` |
| `file.zip` | string | zip file being processed | `"warblrb10k_public_wav.zip"` | `extract.py` |
| `file.csv` | string | csv file being processed | `"warblrb10k_public_metadata_2018.csv"` | `extract.py` |
| `file.dest` | string | destination path for a move | `"data/raw/train/warblrb10k/"` | `extract.py` |
| `file.reason` | string | why something was skipped | `"already extracted"` | `extract.py` |

### Counts

| Name | Type | Description | Example | Used by |
| --- | --- | --- | --- | --- |
| `count.created` | int | newly created count | `16` | `setup.py` |
| `count.existed` | int | already-existing count | `0` | `setup.py`, `preprocess.py` |
| `count.computed` | int | newly computed count | `1230` | `preprocess.py` |
| `count.skipped` | int | skipped count | `45` | `preprocess.py` |
| `count.percent` | float | progress percentage | `40` | `preprocess.py` |

## 3. Log level definitions with examples

Standard used: RFC 5424 (syslog), 8 severity levels. `write_log()` has no level support yet, so this table is forward-looking, not current behaviour.

| Level | Name | Description | Currently active | Example |
| --- | --- | --- | --- | --- |
| 0 | Emergency | system is unusable | No | Starling's main loop can't recover and the device is non-functional |
| 1 | Alert | action must be taken immediately | No | battery critically low, device about to shut down |
| 2 | Critical | critical condition | No | SQLite database is corrupted or unwritable |
| 3 | Error | an operation failed | No | LoRaWAN packet failed to send after retries |
| 4 | Warning | something unexpected, not yet a failure | No | a training run's val loss diverged but didn't crash |
| 5 | Notice | normal but significant event | No | a new baseline was promoted; a training run finished |
| 6 | Informational | routine operational detail | No | "started fold", "completed epoch", most current `write_log()` calls |
| 7 | Debug | detail only useful while developing/diagnosing | No | per-batch tensor shapes, raw intermediate values |

## 4. Common patterns for your domain

*To be filled in at a later date.*

## 5. Anti-patterns to avoid

*To be filled in at a later date.*
