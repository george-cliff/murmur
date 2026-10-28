# stdlib
import copy
import os
import time
import gc

# must be set before torch initialises its CUDA allocator, so it has to come before the import below
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# third-party
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

# local
from sparrowhawk.data import get_manifest, BirdDataset, build_dataloader
from sparrowhawk.model import Sparrowhawk
from sparrowhawk.experiment import (
    run_epoch,
    print_model_banner,
    sweep_thresholds,
    SEED_MAP,
    BATCH_SIZE,
    NUM_WORKERS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    PATIENCE,
    LR_FACTOR,
    SCHED_PATIENCE,
    AUGMENT_NOISE,
    AUGMENT_MASK,
    POS_WEIGHT_MULTI,
    BETA_MAP,
    THRESHOLD_MAP,
    TARGET_BETA,
)
from sparrowhawk.utils import next_run_id
from tools.log import start_logging, write_log
from tools.paths import SPARROWHAWK_MODEL_DATA_DIR, MODELS_DIR, SPARROWHAWK_LOG_DIR

SCOPE = "TRAIN"
SEED = SEED_MAP[0]  # only runs the final training on the first


def train():
    run_start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(SEED)

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

    manifest = get_manifest()
    train_df, val_df = train_test_split(
        manifest,
        test_size=0.15,
        train_size=0.85,
        random_state=SEED % 4294967296,
        shuffle=True,
    )

    counts = train_df["hasbird"].value_counts()
    base_pos_weight = counts[0] / counts[1]
    pos_weight = torch.tensor(base_pos_weight * POS_WEIGHT_MULTI).to(device)
    write_log(SCOPE, "split data", data_name="train", data_rows=len(train_df))
    write_log(SCOPE, "split data", data_name="val", data_rows=len(val_df))

    val_split_path = f"{SPARROWHAWK_MODEL_DATA_DIR}/sparrowhawk-{RUN_ID}_val.csv"
    val_df.to_csv(val_split_path, index=False)
    write_log(SCOPE, "saved val split", file_path=val_split_path)
    write_log(SCOPE, "computed pos weight", model_pos_weight=f"{pos_weight.item():.3f}")

    train_data = BirdDataset(
        df=train_df,
        seed=SEED,
        augment_noise=AUGMENT_NOISE,
        augment_mask=AUGMENT_MASK,
    )
    val_data = BirdDataset(df=val_df)

    train_dataloader = build_dataloader(
        train_data,
        "train",
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
    )
    val_dataloader = build_dataloader(
        val_data,
        "val",
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
    )

    model = Sparrowhawk().to(device)
    print_model_banner(model)

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    scaler = torch.amp.GradScaler(device.type)
    optimiser = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="min", factor=LR_FACTOR, patience=SCHED_PATIENCE
    )

    best_auc = 0.00
    best_model_state = None
    epoch_no_improve = 0
    epoch = 0

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
            epoch_number=epoch,
            metric_auc=f"{epoch_auc:.3f}",
            metric_status=status,
            metric_train_loss=f"{epoch_train_loss:.4f}",
            metric_val_loss=f"{epoch_val_loss:.4f}",
            time_data_seconds=f"{data_time:.1f}",
            time_compute_seconds=f"{compute_time:.1f}",
        )

        lr_before = optimiser.param_groups[0]["lr"]
        scheduler.step(epoch_val_loss)
        lr_after = optimiser.param_groups[0]["lr"]
        if lr_after < lr_before:
            write_log(
                SCOPE,
                "dropped LR",
                metric_lr_before=f"{lr_before:.6f}",
                metric_lr_after=f"{lr_after:.6f}",
            )

        epoch += 1

    write_log(
        SCOPE,
        "reverted to best checkpoint",
        epoch_number=epoch,
        metric_best_auc=f"{best_auc:.3f}",
    )
    model.load_state_dict(best_model_state)

    best_thresholds = sweep_thresholds(
        model, val_dataloader, device, BETA_MAP, THRESHOLD_MAP
    )
    write_log(
        SCOPE,
        "selected threshold",
        metric_beta=TARGET_BETA,
        metric_threshold=f"{best_thresholds[TARGET_BETA]:.2f}",
    )

    save_path = f"{MODELS_DIR}/sparrowhawk-{RUN_ID}.pt"
    torch.save(obj=model.state_dict(), f=save_path)
    write_log(SCOPE, "saved model", file_path=save_path)

    run_elapsed = time.time() - run_start
    write_log(
        SCOPE,
        "finished run",
        time_minutes=f"{run_elapsed / 60:.1f}",
        time_seconds=f"{run_elapsed:.0f}",
    )


if __name__ == "__main__":
    gc.collect()
    torch.cuda.empty_cache()
    RUN_ID = next_run_id("V", SPARROWHAWK_LOG_DIR)
    log_file = start_logging(SPARROWHAWK_LOG_DIR, f"sparrowhawk-{RUN_ID}")
    train()
