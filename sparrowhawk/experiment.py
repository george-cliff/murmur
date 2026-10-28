# stdlib
import copy
import signal
import time

# third-party
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    fbeta_score,
    accuracy_score,
    recall_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn

# local
from sparrowhawk.data import (
    FOLD_MAP,
    FULL_CLIP,
    MAX_MASK_WIDTH,
    NOISE_WEIGHT,
    get_manifest,
    get_fold,
    filter_manifest,
    BirdDataset,
    build_dataloader,
)
from sparrowhawk.model import Sparrowhawk, NEG_SLOPE
from sparrowhawk.utils import next_run_id
from tools.log import start_logging, write_log
from tools.paths import ON_KAGGLE, SPARROWHAWK_LOG_DIR

SCOPE = "EXPERIMENT"


# ============================================================
# SEED MAP
# ============================================================

# seed map to create reproducible runs on a variety of seeds and eliminate
# the chance of testing on a lucky seed | some of these seeds are
# definitely not lucky :(
SEED_MAP = [  # These seeds are all encrypted words related to the project, see if you can crack them (I reccomend avoiding 2,5, and 9 to start with and finishing with 10)
    213243223153243223,
    141232223151155133,
    134124223155,
    143212212225232,
    141231111241144,
    152124223115135,
    143243225232,
    152252232135212155,
    143152142,
    252142214124124,
]

# ============================================================
# BASELINE INFO
# ============================================================

BASELINE_AUC = 0.8368  # harmonic mean auc of the latest baseline, following DCASE scoring method, and stays valid even when loss is altered (e.g changing `pos_weight`)
BASELINE_UNW_LOSS = 0.7132  # mean unweighted test loss of the latest baseline - always computed with pos_weight=1, so stays comparable even when loss is altered (e.g. changing `pos_weight`)
BASELINE_RECALL = 0.6062  # mean recall of the latest baseline, at whatever threshold each seed's F-beta sweep picked

# ============================================================
# RUN CONTROL
# ============================================================

AUGMENT_NOISE = False  # add synthetic green noise at NOISE_WEIGHT (in `data.py`) to drop the signal to noise ratio
AUGMENT_MASK = True  # zero out a random frequency/time band
PATIENCE = 10  # epochs no imrpove before reverting model

# ============================================================
# SMOKE TEST SETTINGS
# ============================================================

# I am currently using a two tier smoke test approach for speeding up
# model training on tuning (I know its not optimal but my ol *RTX 2060*
# cant handle too much)
# tier 1: 3 folds, 3 seeds - tests for has this made a big obvious change
# good or bad
# tier 2: 3 folds, 5-8 seeds - confirmation on the result (more seeds if
# the change was minimal)
SMOKE_TEST = True
SMOKE_TEST_FOLDS = 3
SMOKE_TEST_SEEDS = 3

# ============================================================
# OPTIMISER SETTINGS
# ============================================================

BATCH_SIZE = 48
NUM_WORKERS = (
    4 if ON_KAGGLE else 6
)  # how many lil guys are fetching/prepping the data from disk, you can use `tools/benchmark_workers.py` to find the best number for you
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
TRAIN_PROGRESS_INTERVAL = 100  # log a heartbeat every N training batches, so a long epoch doesn't look stuck

# ============================================================
# LOSS SETTINGS
# ============================================================

# base_pos_weight alone already balances the loss for this fold's
# class imbalance - POS_WEIGHT_MULTI then pushes it further on top
# of that, deliberately biasing the model toward catching more real
# bird calls at the cost of more false alarms. 1 leaves the natural
# balance untouched; raise it to trade precision for recall.

POS_WEIGHT_MULTI = 1  # add a multiplier to pos_weight to push model to higher recall

# ============================================================
# LR SCHEDULER SETTINGS
# ============================================================

LR_FACTOR = 0.5  # multiply LR by this when the scheduler drops it
SCHED_PATIENCE = 5  # epochs with no improvement before dropping

# ============================================================
# THRESHOLD SWEEP SETTINGS
# ============================================================


# BETA_MAP/THRESHOLD_MAP run the validation sweep: every threshold gets tried
# for every beta, best F-beta wins. F-beta trades off missed birds vs false
# alarms (beta > 1 favours fewer misses, more false alarms - vice versa).
# TARGET_BETA then picks which beta's numbers get kept for fold/overall
# summaries - raise it if missed birds should matter more for tracking,
# lower it if false alarms should matter more.


BETA_MAP = np.arange(0.50, 2.01, 0.5)  # f-betas to try during the val sweep
THRESHOLD_MAP = np.arange(0.20, 1.00, 0.05)  # thresholds used during the val sweep
TARGET_BETA = 1.0  # which beta's scores get kept for fold/overall summaries


# ============================================================
# INTERRUPT HANDLING - DO NOT TOUCH UNLESS YOU KNOW WHAT YOU'RE DOING
# ============================================================

# DO NOT CHANGE - NOT A CONSTANT - this is mutable, it gets flipped to
# True by _request_exit() when Ctrl+C is pressed, and checked by the
# training loop each epoch to know when to stop early.
_exit_flag = False

# WARNING: Ctrl+C doesn't kill a run immediately - it just signals an early
# exit. The run finishes the current epoch, reverts to the best checkpoint,
# then exits cleanly handled by _exit_flag above and _request_exit below. If something's
# wrong and you need it dead, you gotta close the terminal (or use
# `kill`/`pkill` on the process).


# handles Ctrl+C for clean exit
def _request_exit(sig, frame):
    global _exit_flag
    _exit_flag = True
    write_log(SCOPE, "received interrupt signal")


# ============================================================
# HELPERS
# ============================================================


# shared by the val pass, threshold sweep and final test pass
def _run_inference(model, dataloader, device):
    with torch.inference_mode():
        all_preds = []
        all_labels = []
        for spect, labels in dataloader:
            spect, labels = spect.to(device), labels.to(device)
            with torch.autocast(device_type=device.type):
                all_preds.append(model(spect))
            all_labels.append(labels)
        return torch.cat(all_preds).squeeze(1), torch.cat(all_labels)


# easy to read fold and overall summary including means and std for tracked metrics
def _print_summary(results, fold=None):
    accs = np.array([r["accuracy"] for r in results])
    recalls = np.array([r["recall"] for r in results])
    aucs = np.array([r["auc"] for r in results])
    fps = np.array([r["false_positives"] for r in results])
    fns = np.array([r["false_negatives"] for r in results])
    total_false = fps + fns
    losses = np.array([r["test_loss"] for r in results])
    unweighted_losses = np.array([r["unweighted_test_loss"] for r in results])
    thresholds = np.array([r["threshold"] for r in results])
    fold_context = {"fold_number": fold} if fold is not None else {}
    if fold is not None:
        banner_text = f"====== FOLD {fold} SUMMARY ======"
    else:
        banner_text = f"====== TRAINING SUMMARY {len(results)} RUNS ======"
    print(banner_text)

    score_metrics = [
        ("Recall", recalls, 3),
        ("AUC", aucs, 3),
        ("Accuracy", accs, 3),
        ("Test loss", losses, 4),
        ("Unw. loss", unweighted_losses, 4),
        ("Threshold", thresholds, 4),
    ]
    for name, array, sf in score_metrics:
        write_log(
            SCOPE,
            "summarised metric",
            **fold_context,
            metric_name=name,
            metric_mean=f"{array.mean():.{sf}f}",
            metric_std=f"{array.std():.{sf}f}",
        )

    print("-" * len(banner_text))

    count_metrics = [
        ("False Pos", fps),
        ("False Neg", fns),
        ("Total False", total_false),
    ]
    for name, array in count_metrics:
        write_log(
            SCOPE,
            "summarised metric",
            **fold_context,
            metric_name=name,
            metric_mean=f"{array.mean():.1f}",
            metric_std=f"{array.std():.1f}",
        )
    print("=" * len(banner_text))


# easy to read model printout replacing default ugly print
def print_model_banner(model):
    banner = "====== MODEL ARCHITECTURE ======"
    print(banner)
    for name, layer in model.named_children():
        layer_type = type(layer).__name__
        if isinstance(layer, nn.Conv2d):
            detail = f"{layer.in_channels} -> {layer.out_channels} channels, kernel {layer.kernel_size[0]}x{layer.kernel_size[1]}"
        elif isinstance(layer, nn.BatchNorm2d):
            detail = f"{layer.num_features} features"
        elif isinstance(layer, nn.Linear):
            detail = f"{layer.in_features} -> {layer.out_features}"
        elif isinstance(layer, nn.Dropout):
            detail = f"p={layer.p}"
        elif isinstance(layer, nn.AdaptiveAvgPool2d):
            detail = f"output {layer.output_size}"
        elif isinstance(layer, nn.Flatten):
            detail = "flattens all dims after the batch dim"
        else:
            detail = str(layer)
        print(f"{name:8s} | {layer_type:14s} | {detail}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters())}")
    print("=" * len(banner))


# ============================================================
# EPOCH LOGIC
# ============================================================


# training and validation for each epoch
def run_epoch(
    model, train_dataloader, val_dataloader, optimiser, loss_fn, device, epoch, scaler
):
    # Training
    model.train()
    train_dataloader.dataset.set_epoch(epoch)  # for aiding in seeding data
    train_loss = 0
    data_time = 0.0
    compute_time = 0.0
    batch_start = time.time()
    batch_total = len(train_dataloader)
    for batch_number, (spect, labels) in enumerate(train_dataloader, start=1):
        data_time += time.time() - batch_start
        compute_start = time.time()
        spect, labels = spect.to(device), labels.to(device)
        # Do the forward pass
        with torch.autocast(device_type=device.type):
            spect_pred = model(spect)
            # Calculate the loss
            loss = loss_fn(spect_pred.squeeze(1), labels.float())
            train_loss += loss.item()
        # Optimiser zero grad
        optimiser.zero_grad()
        # Lossssss backward
        scaler.scale(loss).backward()
        # Optimiser step step step
        scaler.step(optimiser)
        scaler.update()
        compute_time += time.time() - compute_start
        batch_start = time.time()

        if batch_number % TRAIN_PROGRESS_INTERVAL == 0 or batch_number == batch_total:
            write_log(
                SCOPE,
                "processed batch",
                epoch_number=epoch,
                data_batch=batch_number,
                data_batch_total=batch_total,
                metric_running_loss=f"{train_loss / batch_number:.4f}",
            )
    epoch_train_loss = train_loss / len(train_dataloader)

    # Validation
    model.eval()
    all_preds_cat, all_labels_cat = _run_inference(model, val_dataloader, device)
    scores = (
        torch.sigmoid(all_preds_cat).cpu().numpy()
    )  # moved here from model for calculating probs as model now returns raw logits
    labels_np = all_labels_cat.cpu().numpy()
    epoch_auc = roc_auc_score(labels_np, scores)
    epoch_val_loss = loss_fn(all_preds_cat, all_labels_cat.float()).item()

    return epoch_train_loss, epoch_val_loss, epoch_auc, data_time, compute_time


# ============================================================
# EVALUATION
# ============================================================


# re run model over validation set, and for each f-beta find the best threshold
def sweep_thresholds(
    model, val_dataloader, device, betas, candidate_thresholds, fold=None, seed=None
):
    fold_seed_context = {}
    if fold is not None:
        fold_seed_context["fold_number"] = fold
    if seed is not None:
        fold_seed_context["seed_number"] = seed
    model.eval()  # re run evalution for reverted model weights
    val_preds, val_labels_cat = _run_inference(model, val_dataloader, device)
    val_scores = torch.sigmoid(val_preds).cpu().numpy()
    val_labels_np = val_labels_cat.cpu().numpy()

    best_thresholds = {}
    for target_beta in betas:
        best_f_beta = -999.0
        best_thresh_for_beta = 0.5
        for thresh in candidate_thresholds:
            preds_binary = (val_scores > thresh).astype(
                int
            )  # converts True/False to 0/1 to match labels
            f_beta = fbeta_score(
                val_labels_np, preds_binary, beta=target_beta, zero_division=0
            )
            if f_beta > best_f_beta:
                best_f_beta = f_beta
                best_thresh_for_beta = thresh
        best_thresholds[target_beta] = best_thresh_for_beta
        write_log(
            SCOPE,
            "swept threshold",
            **fold_seed_context,
            metric_beta=target_beta,
            metric_threshold=f"{best_thresh_for_beta:.2f}",
            metric_f_beta=f"{best_f_beta:.3f}",
        )
    return best_thresholds


# run final eval on reverted model over test dataset
def _evaluate_test_set(
    model,
    test_dataloader,
    device,
    loss_fn,
    best_thresholds,
    betas,
    fold=None,
    seed=None,
):
    fold_seed_context = {}
    if fold is not None:
        fold_seed_context["fold_number"] = fold
    if seed is not None:
        fold_seed_context["seed_number"] = seed
    model.eval()
    final_preds, final_labels = _run_inference(model, test_dataloader, device)
    final_scores = torch.sigmoid(final_preds).cpu().numpy()
    final_labels_np = final_labels.cpu().numpy()
    final_auc = roc_auc_score(final_labels_np, final_scores)
    test_loss = loss_fn(final_preds, final_labels.float()).item()
    unweighted_test_loss = nn.BCEWithLogitsLoss()(  # treats loss as if pos_weight = 1
        final_preds, final_labels.float()
    ).item()

    # compute every beta's metrics first, print after
    test_result = None
    beta_results = []
    for target_beta in betas:
        b_thresh = best_thresholds[target_beta]
        preds_binary = (final_scores > b_thresh).astype(int)
        accuracy = accuracy_score(final_labels_np, preds_binary)
        recall = recall_score(final_labels_np, preds_binary, zero_division=0)
        # get the false predicitions from the confusion matrix only, as true preds just add noise
        _, false_positives, false_negatives, _ = confusion_matrix(
            final_labels_np, preds_binary, labels=[0, 1]
        ).ravel()
        beta_results.append(
            (target_beta, b_thresh, recall, accuracy, false_positives, false_negatives)
        )

        if target_beta == TARGET_BETA:
            test_result = {
                "accuracy": accuracy,
                "recall": recall,
                "auc": final_auc,
                "false_positives": false_positives,
                "false_negatives": false_negatives,
                "test_loss": test_loss,
                "unweighted_test_loss": unweighted_test_loss,
                "threshold": b_thresh,
            }

    if fold is not None and seed is not None:
        banner_text = f"====== FOLD {fold} SEED {seed} EVAL ======"
    else:
        banner_text = "====== FINAL EVAL ======"
    print(banner_text)

    score_metrics = [
        ("AUC", final_auc, 3),
        ("Test loss", test_loss, 4),
        ("Unw. loss", unweighted_test_loss, 4),
    ]
    for name, value, sf in score_metrics:
        write_log(
            SCOPE,
            "evaluated test set",
            **fold_seed_context,
            metric_name=name,
            metric_value=f"{value:.{sf}f}",
        )

    print("-" * len(banner_text))

    for (
        target_beta,
        b_thresh,
        recall,
        accuracy,
        false_positives,
        false_negatives,
    ) in beta_results:
        write_log(
            SCOPE,
            "evaluated beta",
            **fold_seed_context,
            metric_beta=target_beta,
            metric_threshold=f"{b_thresh:.2f}",
            metric_recall=f"{recall:.3f}",
            metric_accuracy=f"{accuracy:.3f}",
        )
        write_log(
            SCOPE,
            "evaluated beta",
            **fold_seed_context,
            metric_beta=target_beta,
            metric_false_positives=false_positives,
            metric_false_negatives=false_negatives,
            metric_total_false=false_positives + false_negatives,
        )

    print("=" * len(banner_text))
    return test_result


# ============================================================
# MAIN TRAINING LOOP
# ============================================================


def train():
    signal.signal(
        signal.SIGINT, _request_exit
    )  # when SIGINT hits, diverts it to clean exit

    run_start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # ============================================================
    # LOG SETUP AND INITLIASTION
    # ============================================================

    write_log(SCOPE, "detected pytorch version", model_torch_version=torch.__version__)
    if torch.cuda.is_available():
        write_log(
            SCOPE,
            "selected device",
            model_device=device,
            model_gpu=torch.cuda.get_device_name(0),
        )
    else:
        write_log(SCOPE, "selected device", model_device=device)
    write_log(
        SCOPE,
        "configured cudnn for reproducibility",
        model_cudnn_deterministic=torch.backends.cudnn.deterministic,
        model_cudnn_benchmark=torch.backends.cudnn.benchmark,
    )

    manifest = get_manifest()
    interrupted = False  # set up for early exit flag
    all_results = []  # collects one dict of stats per seed, across every fold
    fold_mean_aucs = []  # collects mean AUC per fold, to caculate harmonic mean baseline
    folds_to_run = range(SMOKE_TEST_FOLDS) if SMOKE_TEST else range(len(FOLD_MAP))
    seeds_to_run = SEED_MAP[:SMOKE_TEST_SEEDS] if SMOKE_TEST else SEED_MAP

    print_model = Sparrowhawk()  # throaway model to print architecutre to the logs
    print_model_banner(print_model)
    del print_model

    # other prints for setup section of the log

    write_log(
        SCOPE,
        "set hyperparameters",
        model_learning_rate=LEARNING_RATE,
        model_weight_decay=WEIGHT_DECAY,
        data_batch_size=BATCH_SIZE,
    )
    write_log(
        SCOPE,
        "set hyperparameters",
        model_patience=PATIENCE,
        model_sched_patience=SCHED_PATIENCE,
    )
    write_log(
        SCOPE,
        "set hyperparameters",
        model_lr_factor=LR_FACTOR,
        model_leaky_relu_slope=NEG_SLOPE,
    )
    write_log(
        SCOPE, "set hyperparameters", model_betas=", ".join(str(b) for b in BETA_MAP)
    )
    write_log(SCOPE, "configured clip length", model_full_clip=FULL_CLIP)
    write_log(
        SCOPE,
        "set augmentation",
        model_noise=AUGMENT_NOISE,
        model_noise_weight=NOISE_WEIGHT,
        model_masking=AUGMENT_MASK,
        model_mask_width=MAX_MASK_WIDTH,
    )

    if SMOKE_TEST:
        write_log(
            SCOPE,
            "enabled smoke test",
            model_smoke_test=True,
            fold_total=len(folds_to_run),
            seed_total=len(seeds_to_run),
        )
        write_log(
            SCOPE, "loaded baseline", baseline_metric="auc", baseline_value=BASELINE_AUC
        )
        write_log(
            SCOPE,
            "loaded baseline",
            baseline_metric="unweighted_loss",
            baseline_value=BASELINE_UNW_LOSS,
        )
        write_log(
            SCOPE,
            "loaded baseline",
            baseline_metric="recall",
            baseline_value=BASELINE_RECALL,
        )

    # ============================================================
    # FOLD LOOP
    # ============================================================
    for fold in folds_to_run:
        fold_start = time.time()
        write_log(
            SCOPE, "started fold", fold_number=fold + 1, fold_total=len(folds_to_run)
        )

        # split the manifest into this fold's train/test dataframes
        train_names, test_names = get_fold(fold)
        train_df = filter_manifest(manifest, train_names)
        test_df = filter_manifest(manifest, test_names)

        # create validation dataframe from training dataframe
        # so test dataframe can be tested blind
        train_df, val_df = train_test_split(
            train_df,
            test_size=0.15,
            train_size=0.85,
            random_state=fold,
            shuffle=True,
        )
        write_log(
            SCOPE,
            "split fold data",
            fold_number=fold + 1,
            data_name="train",
            data_rows=len(train_df),
        )
        write_log(
            SCOPE,
            "split fold data",
            fold_number=fold + 1,
            data_name="val",
            data_rows=len(val_df),
        )
        write_log(
            SCOPE,
            "split fold data",
            fold_number=fold + 1,
            data_name="test",
            data_rows=len(test_df),
        )

        # apply pos_weight and POS_WEIGHT_MULTI to the tensor, per fold
        counts = train_df["hasbird"].value_counts()
        base_pos_weight = counts[0] / counts[1]
        pos_weight = torch.tensor(base_pos_weight * POS_WEIGHT_MULTI).to(device)

        write_log(
            SCOPE,
            "computed pos weight",
            fold_number=fold + 1,
            model_pos_weight=f"{pos_weight.item():.3f}",
            model_base_pos_weight=f"{base_pos_weight:.3f}",
            model_pos_weight_multi=POS_WEIGHT_MULTI,
        )

        # val and test data built pre-seed stop unnecessary dataloading
        # val and test data never get augmented so seed and augment passing are irrelevant
        val_data = BirdDataset(df=val_df, fold=fold)
        test_data = BirdDataset(df=test_df, fold=fold)
        val_dataloader = build_dataloader(
            val_data,
            "val",
            fold=fold + 1,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            persistent_workers=True,
        )
        test_dataloader = build_dataloader(
            test_data,
            "test",
            fold=fold + 1,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            persistent_workers=False,
        )

        seed_results = []  # collects dict of stats for per seed

        # ============================================================
        # SEED LOOP
        # ============================================================
        for seed_num, seed in enumerate(seeds_to_run, start=1):
            seed_start = time.time()
            torch.manual_seed(seed)
            write_log(
                SCOPE,
                "started seed",
                fold_number=fold + 1,
                seed_number=seed_num,
                seed_value=seed,
            )

            # train_data rebuilt every seed as it has its own seed separate from global that affects crop/augmentation
            # train_dataloader rebuilt every seed as shuffle order depends on torch.manual_seed(seed) above
            train_data = BirdDataset(
                df=train_df,
                seed=seed,
                fold=fold,
                augment_noise=AUGMENT_NOISE,
                augment_mask=AUGMENT_MASK,
            )
            train_dataloader = build_dataloader(
                train_data,
                "train",
                fold=fold + 1,
                seed=seed_num,
                batch_size=BATCH_SIZE,
                shuffle=True,
                num_workers=NUM_WORKERS,
                pin_memory=True,
                persistent_workers=True,
            )

            # generate a fresh model, loss_fn, optim, and scheduler for the seed
            model = Sparrowhawk().to(device)
            loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            scaler = torch.amp.GradScaler(device.type)
            optimiser = torch.optim.Adam(
                model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
            )
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimiser, mode="min", factor=LR_FACTOR, patience=SCHED_PATIENCE
            )

            # intialise epoch variables pre loop
            best_auc = 0.00
            best_model_state = None
            epoch_no_improve = 0
            epoch = 0

            # ============================================================
            # EPOCH LOOP
            # ============================================================
            while epoch_no_improve < PATIENCE:
                epoch_train_loss, epoch_val_loss, epoch_auc, data_time, compute_time = (
                    run_epoch(
                        model,
                        train_dataloader,
                        val_dataloader,
                        optimiser,
                        loss_fn,
                        device,
                        epoch,
                        scaler,
                    )
                )

                # checkpointing and saving of best model state
                if epoch_auc > best_auc:
                    best_auc = epoch_auc
                    epoch_no_improve = 0
                    best_model_state = copy.deepcopy(model.state_dict())
                    status = "best"
                else:
                    epoch_no_improve += 1
                    status = f"stale {epoch_no_improve}/{PATIENCE}"
                write_log(
                    SCOPE,
                    "completed epoch",
                    fold_number=fold + 1,
                    seed_number=seed_num,
                    epoch_number=epoch,
                    metric_auc=f"{epoch_auc:.3f}",
                    metric_status=status,
                    metric_train_loss=f"{epoch_train_loss:.4f}",
                    metric_val_loss=f"{epoch_val_loss:.4f}",
                    time_data_seconds=f"{data_time:.1f}",
                    time_compute_seconds=f"{compute_time:.1f}",
                )

                # scheduler currently watches val_loss not val_auc, may change
                lr_before = optimiser.param_groups[0]["lr"]
                scheduler.step(epoch_val_loss)
                lr_after = optimiser.param_groups[0]["lr"]
                if lr_after < lr_before:
                    write_log(
                        SCOPE,
                        "dropped LR",
                        fold_number=fold + 1,
                        seed_number=seed_num,
                        metric_lr_before=f"{lr_before:.6f}",
                        metric_lr_after=f"{lr_after:.6f}",
                    )

                # early exit for Ctrl+C press
                if _exit_flag:
                    write_log(
                        SCOPE,
                        "stopped seed early",
                        fold_number=fold + 1,
                        seed_number=seed_num,
                        epoch_number=epoch,
                    )
                    if best_model_state is not None:
                        model.load_state_dict(best_model_state)
                    interrupted = True
                    break
                epoch += 1

            # seed done, revert to best model state
            if not interrupted:
                write_log(
                    SCOPE,
                    "reverted to best checkpoint",
                    fold_number=fold + 1,
                    seed_number=seed_num,
                    epoch_number=epoch,
                    metric_best_auc=f"{best_auc:.3f}",
                )
                model.load_state_dict(best_model_state)

            # ============================================================
            # SEED SUMMARY
            # ============================================================
            best_thresholds = sweep_thresholds(
                model,
                val_dataloader,
                device,
                BETA_MAP,
                THRESHOLD_MAP,
                fold=fold + 1,
                seed=seed_num,
            )

            test_result = _evaluate_test_set(
                model,
                test_dataloader,
                device,
                loss_fn,
                best_thresholds,
                BETA_MAP,
                fold=fold + 1,
                seed=seed_num,
            )
            seed_results.append(test_result)
            all_results.append(test_result)

            seed_elapsed = time.time() - seed_start
            write_log(
                SCOPE,
                "finished seed",
                fold_number=fold + 1,
                seed_number=seed_num,
                time_minutes=f"{seed_elapsed / 60:.1f}",
                time_seconds=f"{seed_elapsed:.0f}",
            )

            if interrupted:
                break

        # ============================================================
        # FOLD SUMMARY
        # ============================================================
        _print_summary(seed_results, fold=fold + 1)
        fold_mean_aucs.append(np.array([r["auc"] for r in seed_results]).mean())
        fold_elapsed = time.time() - fold_start
        write_log(
            SCOPE,
            "finished fold",
            fold_number=fold + 1,
            time_minutes=f"{fold_elapsed / 60:.1f}",
            time_seconds=f"{fold_elapsed:.0f}",
        )
        if interrupted:
            break

    if not all_results:
        run_elapsed = time.time() - run_start
        write_log(SCOPE, "found no completed runs")
        write_log(
            SCOPE,
            "finished run",
            time_minutes=f"{run_elapsed / 60:.1f}",
            time_seconds=f"{run_elapsed:.0f}",
        )
        return

    # ============================================================
    # TRAINING SUMMARY
    # ============================================================
    _print_summary(all_results)

    # Calculate the harminc mean of the 3 folds AUCs (DCASE's scoring method), to punish bad folds
    fold_aucs_arr = np.array(fold_mean_aucs)
    harmonic_auc = len(fold_aucs_arr) / np.sum(1.0 / fold_aucs_arr)
    mean_unweighted_loss = np.array(
        [r["unweighted_test_loss"] for r in all_results]
    ).mean()
    mean_recall = np.array([r["recall"] for r in all_results]).mean()
    beat_baseline = harmonic_auc > BASELINE_AUC
    beat_baseline_loss = mean_unweighted_loss < BASELINE_UNW_LOSS
    beat_baseline_recall = mean_recall > BASELINE_RECALL
    write_log(
        SCOPE,
        "computed harmonic mean",
        fold_total=len(fold_aucs_arr),
        metric_auc=f"{harmonic_auc:.4f}",
        metric_unweighted_loss=f"{mean_unweighted_loss:.4f}",
        metric_recall=f"{mean_recall:.4f}",
    )
    # purely infomation, requires manual update to constant
    write_log(
        SCOPE,
        "compared baseline",
        baseline_metric="auc",
        baseline_value=BASELINE_AUC,
        baseline_beat=beat_baseline,
    )
    write_log(
        SCOPE,
        "compared baseline",
        baseline_metric="unweighted_loss",
        baseline_value=BASELINE_UNW_LOSS,
        baseline_beat=beat_baseline_loss,
    )
    write_log(
        SCOPE,
        "compared baseline",
        baseline_metric="recall",
        baseline_value=BASELINE_RECALL,
        baseline_beat=beat_baseline_recall,
    )

    run_elapsed = time.time() - run_start
    write_log(
        SCOPE,
        "finished run",
        time_minutes=f"{run_elapsed / 60:.1f}",
        time_seconds=f"{run_elapsed:.0f}",
    )


if __name__ == "__main__":  # its time to train
    RUN_ID = next_run_id(
        "K" if ON_KAGGLE else "L", SPARROWHAWK_LOG_DIR
    )  # e.g. "K1" - K/L for Kaggle/local, number auto-counted from existing logs
    log_file = start_logging(SPARROWHAWK_LOG_DIR, f"sparrowhawk-{RUN_ID}")
    train()
