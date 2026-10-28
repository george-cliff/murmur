# stdlib

# third-party
import onnxruntime as ort
import torch
import numpy as np

# local
from sparrowhawk.model import Sparrowhawk
from sparrowhawk.utils import latest_run_id
from tools.log import start_logging, write_log
from tools.paths import MODELS_DIR, SPARROWHAWK_LOG_DIR

SCOPE = "EXPORT"
MODEL_PATH = f"{MODELS_DIR}/sparrowhawk-{latest_run_id('V', MODELS_DIR)}"


def export():
    model = Sparrowhawk()
    model.load_state_dict(torch.load(f"{MODEL_PATH}.pt"))
    model.eval()
    dummy_input = torch.randn(size=(1, 1, 128, 1000))
    with torch.inference_mode():
        pt_output = model(dummy_input)
    torch.onnx.export(
        model=model,
        args=dummy_input,
        f=f"{MODEL_PATH}.onnx",
        input_names=["mel_spectrogram"],
        output_names=["raw_logit"],
        verbose=False
    )
    write_log(SCOPE, "exported onnx", file_path=f"{MODEL_PATH}.onnx")
    session = ort.InferenceSession(f"{MODEL_PATH}.onnx")
    onnx_output = session.run(
        output_names=None,
        input_feed={"mel_spectrogram": dummy_input.numpy()},  # None returns all outputs
    )
    max_diff = np.abs(pt_output.numpy() - onnx_output[0]).max()
    if np.allclose(pt_output.numpy(), onnx_output[0]):
        write_log(SCOPE, "succeeded onnx export verification", file_path=f"{MODEL_PATH}.onnx", file_max_diff=f"{max_diff:.6f}")
    else:
        write_log(SCOPE, "failed onnx export verification", file_path=f"{MODEL_PATH}.onnx", file_max_diff=f"{max_diff:.6f}")


if __name__ == "__main__":
    start_logging(SPARROWHAWK_LOG_DIR, "export")
    export()
