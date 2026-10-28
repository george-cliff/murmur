use ndarray::{Array2, Array4, Axis, s};
use ort::session::Session;
use ort::value::Tensor;
use std::fs::read_to_string;
use std::path::Path;

use crate::{
    audio::{audio_pipeline, spect_pipeline},
    model,
};

pub fn load_threshold(path: &Path) -> Result<f32, Box<dyn std::error::Error>> {
    let content = read_to_string(path)?;
    let threshold = content.trim().parse::<f32>()?;
    Ok(threshold)
}

fn fit_to_target(mel_log: Array2<f32>, target: usize) -> Array2<f32> {
    if mel_log.ncols() > target {
        let sliced_mel = mel_log.slice(s![.., ..target]).to_owned();
        sliced_mel
    } else {
        let (rows, cols) = mel_log.dim();
        let mut padded_mel = Array2::zeros((rows, target));
        padded_mel.slice_mut(s![.., ..cols]).assign(&mel_log);
        padded_mel
    }
}

fn to_input_array(fitted_mel: Array2<f32>) -> Array4<f32> {
    let input_array = fitted_mel.insert_axis(Axis(0)).insert_axis(Axis(0));
    input_array
}

fn to_tensor(input_array: Array4<f32>) -> Result<Tensor<f32>, Box<dyn std::error::Error>> {
    let tensor = Tensor::from_array(input_array)?;
    Ok(tensor)
}

pub fn detect(
    wav_path: &Path,
    session: &mut Session,
    threshold: f32,
) -> Result<(bool, f32), Box<dyn std::error::Error>> {
    let samples = audio_pipeline(wav_path)?;
    let spect_mel = spect_pipeline(&samples)?;
    let spect_fitted = fit_to_target(spect_mel, 1000);
    let spect_array = to_input_array(spect_fitted);
    let spect_tensor = to_tensor(spect_array)?;
    let spect_logit =
        model::run_inference(session, spect_tensor, "mel_spectrogram", "raw_logit")?[0];
    let spect_sigmoid = model::sigmoid(spect_logit);
    let is_bird = spect_sigmoid > threshold;
    Ok((is_bird, spect_sigmoid))
}
