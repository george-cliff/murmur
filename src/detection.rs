use std::path::Path;

use rusqlite::Connection;

use crate::{db, model, perch, sparrowhawk};

pub fn insert_detection(
    conn: &Connection,
    unit_id: &str,
    deployment_id: i64,
    sequence_no: i64,
    model_version_id: i64,
    model_label_id: i64,
    confidence: f32,
) -> Result<(), Box<dyn std::error::Error>> {
    conn.execute(
        "INSERT OR IGNORE INTO detections
            (unit_id, deployment_id, sequence_no, detected_at, model_version_id, model_label_id, confidence)
         VALUES (?1, ?2, ?3, datetime('now'), ?4, ?5, ?6)",
        rusqlite::params![
            unit_id,
            deployment_id,
            sequence_no,
            model_version_id,
            model_label_id,
            confidence
        ],
    )?;
    Ok(())
}

const MODEL_PATH: &str = "models/sparrowhawk-V9Q.onnx";
const THRESHOLD_PATH: &str = "data/model_data/sparrowhawk/sparrowhawk-V9_threshold.txt";
const PERCH_MODEL_PATH: &str = "models/perch_v2.onnx";
const PERCH_LABELS_PATH: &str = "data/model_data/perch/perch-v2-labels.csv";

pub const WAV_MAP: [&str; 6] = [
    "data/sparrowhawk/raw/train/warblrb10k/wav/759808e5-f824-401e-9058.wav",
    "data/sparrowhawk/raw/train/freefield1010/wav/19037.wav",
    "data/sparrowhawk/raw/train/birdvox/wav/00053d90-e4b9-4045-a2f1-f39efc90cfa9.wav",
    "data/sparrowhawk/raw/train/warblrb10k/wav/ab322d4b-da69-4b06-a065.wav",
    "data/sparrowhawk/raw/train/freefield1010/wav/64486.wav",
    "data/sparrowhawk/raw/train/birdvox/wav/000db435-a40f-4ad9-a74e-d1af284d2c44.wav",
];

pub fn run_models(conn: &Connection, unit_id: &str, deployment_id: i64) {
    let mut session_sparrowhawk = model::load_model(Path::new(MODEL_PATH)).unwrap();
    let threshold = sparrowhawk::load_threshold(Path::new(THRESHOLD_PATH)).unwrap();
    let mut session_perch = model::load_model(Path::new(PERCH_MODEL_PATH)).unwrap();
    let labels = perch::load_labels(Path::new(PERCH_LABELS_PATH)).unwrap();

    let model_version_id = db::seed_perch_model_version(conn).unwrap();

    for (wav_idx, wav_file) in WAV_MAP.iter().enumerate() {
        let (is_bird, confidence) =
            sparrowhawk::detect(Path::new(wav_file), &mut session_sparrowhawk, threshold).unwrap();
        println!(
            "{}: {:.3}",
            if is_bird { "bird" } else { "not bird" },
            confidence
        );
        if is_bird {
            let results =
                perch::identify_species(Path::new(wav_file), &mut session_perch, &labels).unwrap();
            for (chunk_idx, (species_name, species_confidence, species_class_idx)) in
                results.iter().enumerate()
            {
                println!(
                    "  chunk {}: {} ({:.3})",
                    chunk_idx, species_name, species_confidence
                );

                let model_label_id = db::get_or_create_model_label(
                    conn,
                    model_version_id,
                    *species_class_idx,
                    species_name,
                )
                .unwrap();

                let sequence_no = (wav_idx * 2 + chunk_idx) as i64;
                insert_detection(
                    conn,
                    unit_id,
                    deployment_id,
                    sequence_no,
                    model_version_id,
                    model_label_id,
                    *species_confidence,
                )
                .unwrap();
            }
        }
    }
}
