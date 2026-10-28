use std::path::Path;

use hound::WavReader;
use image;
use ndarray::Array2;
use rubato::audioadapter_buffers::direct::InterleavedSlice;
use rubato::{Fft, FixedSync, Resampler};
use spectrs::spectrogram::mel::{MelScale, convert_to_mel};
use spectrs::spectrogram::stft::{SpectrogramType, compute_spectrogram};

// Constants derived from Perch spectrogram parameters - https://github.com/google-research/perch
const FRAME_SIZE: usize = 1024;
const WIN_SIZE: usize = 640;
const HOP_SIZE: usize = 320;
const N_MELS: u32 = 128;
const F_MIN: f32 = 60.0;
const F_MAX: f32 = 16000.0;
const TARGET_SAMPLE_RATE: usize = 32000;

fn load_wav(wav_file: &Path) -> Result<(Vec<f32>, u32), Box<dyn std::error::Error>> {
    let mut reader = WavReader::open(wav_file)?;
    let sample_rate = reader.spec().sample_rate;
    let samples: Vec<f32> = reader
        .samples::<i16>()
        .collect::<Result<Vec<i16>, _>>()?
        .into_iter()
        .map(|s| s as f32 / i16::MAX as f32)
        .collect();
    Ok((samples, sample_rate))
}

pub fn normalise_mel(mel_log: Array2<f32>) -> Array2<f32> {
    let min_val = mel_log.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_val = mel_log.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    if max_val == min_val {
        let mel_norm = Array2::zeros(mel_log.dim());
        return mel_norm;
    }
    let mel_norm = mel_log.mapv(|v| (v - min_val) / (max_val - min_val)); // add * 255.0 for png

    mel_norm
}

fn resample_to_32k(
    samples: &[f32],
    source_rate: u32,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let channels = 1; // mono audio

    let mut resampler = Fft::<f32>::new(
        source_rate as usize,
        TARGET_SAMPLE_RATE,
        1024,
        1,
        channels,
        FixedSync::Both,
    )?;

    let input_frames = samples.len();
    let input_adapter = InterleavedSlice::new(samples, channels, input_frames)?;

    let output_len = resampler.process_all_needed_output_len(input_frames);
    let mut output = vec![0.0f32; output_len];
    let mut output_adapter = InterleavedSlice::new_mut(&mut output, channels, output_len)?;

    let (_in_frames, out_frames) = resampler.process_all_into_buffer(
        &input_adapter,
        &mut output_adapter,
        input_frames,
        None,
    )?;
    output.truncate(out_frames);

    Ok(output)
}

pub fn audio_pipeline(wav_file: &Path) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let (samples, sample_rate) = load_wav(wav_file)?;
    let samples_32k = resample_to_32k(&samples, sample_rate)?;
    Ok(samples_32k)
}

pub fn spect_pipeline(samples: &[f32]) -> Result<Array2<f32>, Box<dyn std::error::Error>> {
    // run Short-Time Fourier Transform
    let stft = compute_spectrogram(
        samples,
        FRAME_SIZE,
        HOP_SIZE,
        WIN_SIZE,
        false,
        SpectrogramType::Magnitude,
    );
    // apply mel filter bank
    let mel = convert_to_mel(
        &stft,
        TARGET_SAMPLE_RATE as u32,
        FRAME_SIZE,
        N_MELS as usize,
        Some(F_MIN),
        Some(F_MAX),
        MelScale::HTK,
    );
    // convert to 2d array (Array2)
    let rows = mel.len();
    let cols = mel[0].len();
    let mel_flat = mel.into_iter().flatten().collect();
    let mel_2d = Array2::from_shape_vec((rows, cols), mel_flat)?;
    let mel_log = mel_2d.mapv(|v| v.max(1e-5_f32).ln() * 0.1);

    Ok(mel_log)
}

pub fn generate_mel_png(
    mel_norm: Array2<f32>,
    output_path: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let (h, w) = mel_norm.dim();

    let mut img = image::GrayImage::new(w as u32, h as u32);
    for y in 0..h {
        for x in 0..w {
            let pixel = image::Luma([mel_norm[[y, x]] as u8]);
            img.put_pixel(x as u32, y as u32, pixel);
        }
    }

    img.save(output_path)?;
    println!("Spectrogram saved to {:?}", output_path);
    Ok(())
}
