use std::path::Path;

use ort::{session::Session, value::Tensor};

pub fn load_model(model_path: &Path) -> Result<Session, Box<dyn std::error::Error>> {
    let session = Session::builder()?.commit_from_file(model_path)?;
    Ok(session)
}
pub fn sigmoid(logit: f32) -> f32 {
    let sigmoid = 1.0 / (1.0 + (-logit).exp());
    sigmoid
}

pub fn run_inference(
    session: &mut Session,
    tensor: Tensor<f32>,
    input_name: &str,
    output_name: &str,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let outputs = session.run(ort::inputs![input_name => tensor])?;
    let (_shape, data) = outputs[output_name].try_extract_tensor::<f32>()?;
    Ok(data.to_vec())
}
