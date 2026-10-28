use moontool::moon::{ForDateTime, MoonPhase};

#[derive(serde::Serialize)]
struct MoonReading {
    phase: String,
    illumination_pct: f64,
    age_days: f64,
    distance_km: f64,
}

pub fn fetch_moon_phase() -> Result<serde_json::Value, Box<dyn std::error::Error>> {
    let moon = MoonPhase::now();
    let reading = MoonReading {
        phase: moon.phase.name().to_string(),
        illumination_pct: moon.fraction_illuminated * 100.0,
        age_days: moon.age,
        distance_km: moon.distance_to_earth_km,
    };
    let json = serde_json::to_value(&reading)?;
    Ok(json)
}
