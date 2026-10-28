# stdlib

# third-party
import numpy as np
from sklearn.metrics import fbeta_score, roc_auc_score

# local
from sparrowhawk.data import BirdDataset, build_dataloader
from sparrowhawk.experiment import BETA_MAP, THRESHOLD_MAP, TARGET_BETA
from sparrowhawk.quantise import load_val_split, run_onnx_inference, MODEL_PATH, VAL_BATCH_SIZE
from sparrowhawk.utils import latest_run_id
from tools.log import start_logging, write_log
from tools.paths import SPARROWHAWK_MODEL_DATA_DIR, MODELS_DIR, SPARROWHAWK_LOG_DIR

SCOPE = "EVAL"
RUN_ID = latest_run_id("V", MODELS_DIR)
QUANT_PATH = f"{MODEL_PATH}Q.onnx"
THRESHOLD_PATH = f"{SPARROWHAWK_MODEL_DATA_DIR}/sparrowhawk-{RUN_ID}_threshold.txt"


def sweep_thresholds(scores, labels):
    """Mirrors experiment.py's sweep_thresholds, but against ONNX scores instead of a live torch model."""
    best_thresholds = {}
    for beta in BETA_MAP:
        best_f_beta = -999.0
        best_thresh = 0.5
        for thresh in THRESHOLD_MAP:
            preds = (scores > thresh).astype(int)
            f_beta = fbeta_score(labels, preds, beta=beta, zero_division=0)
            if f_beta > best_f_beta:
                best_f_beta = f_beta
                best_thresh = thresh
        best_thresholds[beta] = best_thresh
        write_log(
            SCOPE,
            "swept threshold",
            metric_beta=beta,
            metric_threshold=f"{best_thresh:.2f}",
            metric_f_beta=f"{best_f_beta:.3f}",
        )
    return best_thresholds


if __name__ == "__main__":
    start_logging(SPARROWHAWK_LOG_DIR, "eval")

    val_df = load_val_split()
    val_data = BirdDataset(df=val_df, seed=None, fold=None)
    val_dataloader = build_dataloader(val_data, "val", batch_size=VAL_BATCH_SIZE, shuffle=False)

    logits, labels = run_onnx_inference(QUANT_PATH, val_dataloader)
    scores = 1 / (1 + np.exp(-logits))  # sigmoid: needed here since thresholds are real probabilities, not just the 0.5 midpoint

    auc = roc_auc_score(labels, logits)  # AUC only cares about ranking, so raw logits are fine here
    write_log(SCOPE, "evaluated quantised model", file_path=QUANT_PATH, metric_auc=f"{auc:.4f}")

    best_thresholds = sweep_thresholds(scores, labels)
    chosen_threshold = best_thresholds[TARGET_BETA]
    write_log(
        SCOPE,
        "selected threshold",
        metric_beta=TARGET_BETA,
        metric_threshold=f"{chosen_threshold:.2f}",
    )

    with open(THRESHOLD_PATH, "w") as f:
        f.write(f"{chosen_threshold:.4f}")
    write_log(SCOPE, "saved threshold", file_path=THRESHOLD_PATH)
