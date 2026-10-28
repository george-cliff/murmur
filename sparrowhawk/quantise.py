# stdlib
import os
import time

# third-party
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
import numpy as np
import onnxruntime as ort
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

# local
from sparrowhawk.data import get_manifest, BirdDataset, build_dataloader
from sparrowhawk.utils import latest_run_id
from tools.log import start_logging, write_log
from tools.paths import SPARROWHAWK_MODEL_DATA_DIR, MODELS_DIR, SPARROWHAWK_LOG_DIR

SCOPE = "QUANTISE"
RUN_ID = latest_run_id("V", MODELS_DIR)
MODEL_PATH = f"{MODELS_DIR}/sparrowhawk-{RUN_ID}"
VAL_SPLIT_PATH = f"{SPARROWHAWK_MODEL_DATA_DIR}/sparrowhawk-{RUN_ID}_val.csv"

CALIBRATION_SAMPLE_SIZE = 1000
CALIBRATION_SEED = 42
# the exported onnx model has a fixed batch dimension of 1 (no dynamic_axes set in export.py)
CALIBRATION_BATCH_SIZE = 1
VAL_BATCH_SIZE = 1


def load_val_split():
    return pd.read_csv(VAL_SPLIT_PATH)


def build_calibration_sample(val_df):
    manifest = get_manifest()
    calib_pool = manifest[~manifest["filepath"].isin(val_df["filepath"])] # get everything from the manifest not in the val_df
    calib_df = calib_pool.sample(n=CALIBRATION_SAMPLE_SIZE, random_state=CALIBRATION_SEED)
    write_log(
        SCOPE,
        "built calibration sample",
        data_rows=len(calib_df),
        data_pool_rows=len(calib_pool),
    )
    return calib_df


# wraps a dataloader so quantize_static can pull one batch at a time via get_next()
class SparrowhawkCalibrationReader(CalibrationDataReader):
    def __init__(self, dataloader):
        self.iterator = iter(dataloader)

    def get_next(self):
        try:
            inputs, _ = next(self.iterator)
        except StopIteration:
            return None
        return {"mel_spectrogram": inputs.numpy()}


def build_calibration_reader(calib_df):
    calib_data = BirdDataset(df=calib_df, seed=None, fold=None)
    calib_dataloader = build_dataloader(
        calib_data,
        "calibration",
        batch_size=CALIBRATION_BATCH_SIZE,
        shuffle=False,
    )
    return SparrowhawkCalibrationReader(calib_dataloader)


def file_size_on_disk(path):
    """Large exports split weights into a companion `.data` file alongside the graph - count both."""
    total = os.path.getsize(path)
    data_path = f"{path}.data"
    if os.path.exists(data_path):
        total += os.path.getsize(data_path)
    return total


def quantise_model(calibration_reader):
    onnx_path = f"{MODEL_PATH}.onnx"
    quant_path = f"{MODEL_PATH}Q.onnx"

    quantize_static(
        model_input=onnx_path,
        model_output=quant_path,
        calibration_data_reader=calibration_reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
    )

    onnx_size = file_size_on_disk(onnx_path)
    quant_size = file_size_on_disk(quant_path)
    write_log(
        SCOPE,
        "quantised model",
        file_path=quant_path,
        data_onnx_bytes=onnx_size,
        data_quant_bytes=quant_size,
        data_size_reduction_pct=f"{(1 - quant_size / onnx_size) * 100:.1f}%",
    )
    return quant_path


def run_onnx_inference(model_path, val_dataloader):
    session = ort.InferenceSession(model_path)
    all_logits = []
    all_labels = []
    for inputs, labels in val_dataloader:
        outputs = session.run(None, {"mel_spectrogram": inputs.numpy()})[0]
        all_logits.append(outputs.item())
        all_labels.append(labels.item())
    return np.array(all_logits), np.array(all_labels)


def compare_accuracy(onnx_path, quant_path, val_df):
    val_data = BirdDataset(df=val_df, seed=None, fold=None)
    val_dataloader = build_dataloader(val_data, "val", batch_size=VAL_BATCH_SIZE, shuffle=False)

    for model_path, variant in [(onnx_path, "float32"), (quant_path, "quantized")]:
        start_time = time.perf_counter()
        logits, labels = run_onnx_inference(model_path, val_dataloader)
        elapsed = time.perf_counter() - start_time
        avg_time_ms = (elapsed / len(labels)) * 1000

        preds = (logits > 0.0).astype(int)  # logit > 0.0 is the same call as sigmoid(logit) > 0.5
        auc = roc_auc_score(labels, logits)  # sigmoid is monotonic, so raw logits rank identically to probabilities
        acc = accuracy_score(labels, preds)
        write_log(
            SCOPE,
            "evaluated model",
            model_variant=variant,
            metric_auc=f"{auc:.4f}",
            metric_accuracy=f"{acc * 100:.2f}%",
            time_avg_ms=f"{avg_time_ms:.2f}",
        )


if __name__ == "__main__":
    start_logging(SPARROWHAWK_LOG_DIR, "quantise")

    val_df = load_val_split()
    calib_df = build_calibration_sample(val_df)
    calibration_reader = build_calibration_reader(calib_df)

    quant_path = quantise_model(calibration_reader)
    compare_accuracy(f"{MODEL_PATH}.onnx", quant_path, val_df)
