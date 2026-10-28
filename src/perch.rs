use std::{fs::read_to_string, path::Path};

use crate::{audio::audio_pipeline, model};
use ndarray::Array2;
use ort::{session::Session, value::Tensor};

pub fn load_labels(path: &Path) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    let content = read_to_string(path)?;
    let labels = content.lines().skip(1).map(|s| s.to_string()).collect();
    Ok(labels)
}

const PERCH_CHUNK_SAMPLES: usize = 160_000; // 5s @ 32kHz

fn get_chunk(samples: &[f32], start: usize, chunk_size: usize) -> Vec<f32> {
    if (samples.len() - start) > chunk_size {
        let chunk = samples[start..start + chunk_size].to_vec();
        chunk
    } else {
        let mut padded_chunk = vec![0.0f32; chunk_size];
        let available = samples.len().saturating_sub(start);
        padded_chunk[..available].copy_from_slice(&samples[start..]);
        padded_chunk
    }
}

fn to_input_array(chunk: Vec<f32>) -> Result<Array2<f32>, Box<dyn std::error::Error>> {
    let chunk_array = Array2::from_shape_vec((1, PERCH_CHUNK_SAMPLES), chunk)?;
    Ok(chunk_array)
}

fn to_tensor(input_array: Array2<f32>) -> Result<Tensor<f32>, Box<dyn std::error::Error>> {
    let tensor = Tensor::from_array(input_array)?;
    Ok(tensor)
}

fn top_prediction(scores: &[f32], labels: &[String]) -> (String, f32, usize) {
    let (top_idx, top_logit) = scores
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap();
    let confidence = model::sigmoid(*top_logit);
    let species = labels[top_idx].clone();
    (species, confidence, top_idx)
}

pub fn identify_species(
    wav_path: &Path,
    session: &mut Session,
    labels: &[String],
) -> Result<Vec<(String, f32, usize)>, Box<dyn std::error::Error>> {
    let samples = audio_pipeline(wav_path)?;
    let mut results = Vec::new();

    for start in [0, PERCH_CHUNK_SAMPLES] {
        let chunk = get_chunk(&samples, start, PERCH_CHUNK_SAMPLES);
        let chunk_array = to_input_array(chunk)?;
        let chunk_tensor = to_tensor(chunk_array)?;
        let scores = model::run_inference(session, chunk_tensor, "inputs", "label")?;
        let (species, confidence, class_idx) = top_prediction(&scores, labels);
        results.push((species, confidence, class_idx));
    }

    Ok(results)
}
