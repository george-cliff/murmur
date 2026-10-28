use std::path::Path;

mod audio;
mod db;
mod detection;
mod model;
mod moon;
mod open_meteo;
mod perch;
mod sparrowhawk;
mod unit;

fn generate_pngs() {
    for wav_file in detection::WAV_MAP {
        let path = Path::new(wav_file);
        let dataset_name = path
            .components()
            .rev()
            .nth(2)
            .unwrap()
            .as_os_str()
            .to_str()
            .unwrap();
        let file_name = path.file_stem().unwrap().to_str().unwrap();
        let output_path = format!("output/spect-{}-{}.png", dataset_name, file_name);
        let _ = std::fs::create_dir_all("output");
        let samples = audio::audio_pipeline(Path::new(wav_file)).unwrap();
        let mel_log = audio::spect_pipeline(&samples).unwrap();
        let mel_norm = audio::normalise_mel(mel_log);
        println!("shape: {:?}", mel_norm.dim());
        let mel_png = mel_norm.mapv(|v| v * 255.0);
        audio::generate_mel_png(mel_png, Path::new(&output_path)).unwrap();
    }
}

fn get_enviroment() {
    let (current_weather, daily_weather) = open_meteo::fetch_weather().unwrap();
    let current_air_quality = open_meteo::fetch_air_quality().unwrap();
    let moon_phase = moon::fetch_moon_phase().unwrap();

    let combined = serde_json::json!({
        "current_weather": current_weather,
        "daily_weather": daily_weather,
        "current_air_quality": current_air_quality,
        "moon_phase": moon_phase,
    });

    std::fs::create_dir_all("output").unwrap();
    std::fs::write(
        "output/environment.json",
        serde_json::to_string_pretty(&combined).unwrap(),
    )
    .unwrap();
}

fn register_and_deploy_unit(conn: &rusqlite::Connection) -> i64 {
    unit::register_unit(conn, unit::TEST_UNIT_ID, unit::TEST_UNIT_SERIAL).unwrap();
    let started_at: String = conn
        .query_row("SELECT datetime('now')", [], |row| row.get(0))
        .unwrap();
    unit::deploy_unit(
        conn,
        unit::TEST_UNIT_ID,
        &started_at,
        unit::TEST_LAT,
        unit::TEST_LONG,
    )
    .unwrap()
}

fn main() {
    println!("initliaising database");
    let conn = db::init_db().unwrap();

    println!("registering and deploying test unit");
    let deployment_id = register_and_deploy_unit(&conn);

    println!("getting environment");
    get_enviroment();

    println!("Generating PNGs");
    generate_pngs();

    println!("Running models");
    detection::run_models(&conn, unit::TEST_UNIT_ID, deployment_id);
}
