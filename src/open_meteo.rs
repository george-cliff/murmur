use reqwest::Url;

const LAT: f64 = crate::unit::TEST_LAT;
const LONG: f64 = crate::unit::TEST_LONG;
const TZ: &str = "Europe/London";
const MODEL: &str = "ukmo_seamless";
const WEATHER_BASE_URL: &str = "https://api.open-meteo.com/v1/forecast";
const AIR_BASE_URL: &str = "https://air-quality-api.open-meteo.com/v1/air-quality";

fn weather_url() -> Result<Url, Box<dyn std::error::Error>> {
    build_url(
        WEATHER_BASE_URL,
        &[
            ("latitude", &LAT.to_string()),
            ("longitude", &LONG.to_string()),
            (
                "daily",
                "sunrise,sunset,weather_code,temperature_2m_max,temperature_2m_min,apparent_temperature_max,apparent_temperature_min,wind_speed_10m_max,wind_gusts_10m_max,wind_direction_10m_dominant,shortwave_radiation_sum,et0_fao_evapotranspiration,daylight_duration,sunshine_duration,uv_index_max,uv_index_clear_sky_max,rain_sum,showers_sum,snowfall_sum,precipitation_sum,precipitation_hours,precipitation_probability_max",
            ),
            ("models", MODEL),
            (
                "current",
                "temperature_2m,precipitation,wind_speed_10m,wind_direction_10m,wind_gusts_10m,relative_humidity_2m,weather_code,rain,showers,snowfall,cloud_cover,pressure_msl,surface_pressure,apparent_temperature,is_day",
            ),
            ("timezone", TZ),
            ("forecast_days", "1"),
        ],
    )
}

fn air_url() -> Result<Url, Box<dyn std::error::Error>> {
    build_url(
        AIR_BASE_URL,
        &[
            ("latitude", &LAT.to_string()),
            ("longitude", &LONG.to_string()),
            (
                "current",
                "european_aqi,us_aqi,pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone,aerosol_optical_depth,dust,uv_index,uv_index_clear_sky,ammonia,alder_pollen,birch_pollen,mugwort_pollen,grass_pollen,olive_pollen,ragweed_pollen",
            ),
            ("forecast_days", "1"),
        ],
    )
}

fn build_url(base: &str, params: &[(&str, &str)]) -> Result<Url, Box<dyn std::error::Error>> {
    let mut url = Url::parse(base)?;
    for (key, value) in params {
        url.query_pairs_mut().append_pair(key, value);
    }
    Ok(url)
}

fn fetch_json(url: Url) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
    let json: serde_json::Value = reqwest::blocking::get(url)?.json()?;
    Ok(json)
}

#[derive(serde::Serialize)]
struct CurrentWeather {
    recorded_at: String,
    temperature_c: f64,
    apparent_temperature_c: f64,
    humidity_pct: f64,
    pressure_hpa: f64,
    surface_pressure_hpa: f64,
    wind_speed_kmh: f64,
    wind_gusts_kmh: f64,
    wind_direction_deg: i64,
    precipitation_mm: f64,
    rain_mm: f64,
    showers_mm: Option<f64>,
    snowfall_cm: f64,
    cloud_cover_pct: f64,
    weather_code: i64,
    is_day: bool,
}

#[derive(serde::Serialize)]
struct DailyWeather {
    observation_date: String,
    sunrise_time: String,
    sunset_time: String,
    temperature_max_c: f64,
    temperature_min_c: f64,
    apparent_temperature_max_c: f64,
    apparent_temperature_min_c: f64,
    precipitation_sum_mm: f64,
    rain_sum_mm: f64,
    showers_sum_mm: Option<f64>,
    snowfall_sum_cm: f64,
    precipitation_hours: f64,
    precipitation_probability_max_pct: Option<f64>,
    wind_speed_max_kmh: f64,
    wind_gusts_max_kmh: f64,
    wind_direction_dominant_deg: i64,
    shortwave_radiation_sum_mjm2: f64,
    et0_fao_evapotranspiration_mm: f64,
    daylight_duration_s: f64,
    sunshine_duration_s: f64,
    uv_index_max: f64,
    uv_index_clear_sky_max: Option<f64>,
    weather_code: i64,
}

#[derive(serde::Serialize)]
struct CurrentAirQuality {
    recorded_at: String,
    european_aqi: i64,
    us_aqi: i64,
    pm10_ugm3: f64,
    pm2_5_ugm3: f64,
    carbon_monoxide_ugm3: f64,
    nitrogen_dioxide_ugm3: f64,
    sulphur_dioxide_ugm3: f64,
    ozone_ugm3: f64,
    aerosol_optical_depth: f64,
    dust_ugm3: f64,
    uv_index: f64,
    uv_index_clear_sky: f64,
    ammonia_ugm3: f64,
    alder_pollen_grainsm3: f64,
    birch_pollen_grainsm3: f64,
    mugwort_pollen_grainsm3: f64,
    grass_pollen_grainsm3: f64,
    olive_pollen_grainsm3: f64,
    ragweed_pollen_grainsm3: f64,
}

fn parse_current_weather(raw: &serde_json::Value) -> CurrentWeather {
    let c = &raw["current"];
    CurrentWeather {
        recorded_at: c["time"].as_str().unwrap().to_string(),
        temperature_c: c["temperature_2m"].as_f64().unwrap(),
        apparent_temperature_c: c["apparent_temperature"].as_f64().unwrap(),
        humidity_pct: c["relative_humidity_2m"].as_f64().unwrap(),
        pressure_hpa: c["pressure_msl"].as_f64().unwrap(),
        surface_pressure_hpa: c["surface_pressure"].as_f64().unwrap(),
        wind_speed_kmh: c["wind_speed_10m"].as_f64().unwrap(),
        wind_gusts_kmh: c["wind_gusts_10m"].as_f64().unwrap(),
        wind_direction_deg: c["wind_direction_10m"].as_i64().unwrap(),
        precipitation_mm: c["precipitation"].as_f64().unwrap(),
        rain_mm: c["rain"].as_f64().unwrap(),
        showers_mm: c["showers"].as_f64(),
        snowfall_cm: c["snowfall"].as_f64().unwrap(),
        cloud_cover_pct: c["cloud_cover"].as_f64().unwrap(),
        weather_code: c["weather_code"].as_i64().unwrap(),
        is_day: c["is_day"].as_i64().unwrap() == 1,
    }
}

fn parse_daily_weather(raw: &serde_json::Value) -> DailyWeather {
    let d = &raw["daily"];
    let first = |field: &str| -> &serde_json::Value { &d[field][0] };
    DailyWeather {
        observation_date: first("time").as_str().unwrap().to_string(),
        sunrise_time: first("sunrise").as_str().unwrap().to_string(),
        sunset_time: first("sunset").as_str().unwrap().to_string(),
        temperature_max_c: first("temperature_2m_max").as_f64().unwrap(),
        temperature_min_c: first("temperature_2m_min").as_f64().unwrap(),
        apparent_temperature_max_c: first("apparent_temperature_max").as_f64().unwrap(),
        apparent_temperature_min_c: first("apparent_temperature_min").as_f64().unwrap(),
        precipitation_sum_mm: first("precipitation_sum").as_f64().unwrap(),
        rain_sum_mm: first("rain_sum").as_f64().unwrap(),
        showers_sum_mm: first("showers_sum").as_f64(),
        snowfall_sum_cm: first("snowfall_sum").as_f64().unwrap(),
        precipitation_hours: first("precipitation_hours").as_f64().unwrap(),
        precipitation_probability_max_pct: first("precipitation_probability_max").as_f64(),
        wind_speed_max_kmh: first("wind_speed_10m_max").as_f64().unwrap(),
        wind_gusts_max_kmh: first("wind_gusts_10m_max").as_f64().unwrap(),
        wind_direction_dominant_deg: first("wind_direction_10m_dominant").as_i64().unwrap(),
        shortwave_radiation_sum_mjm2: first("shortwave_radiation_sum").as_f64().unwrap(),
        et0_fao_evapotranspiration_mm: first("et0_fao_evapotranspiration").as_f64().unwrap(),
        daylight_duration_s: first("daylight_duration").as_f64().unwrap(),
        sunshine_duration_s: first("sunshine_duration").as_f64().unwrap(),
        uv_index_max: first("uv_index_max").as_f64().unwrap(),
        uv_index_clear_sky_max: first("uv_index_clear_sky_max").as_f64(),
        weather_code: first("weather_code").as_i64().unwrap(),
    }
}

fn parse_current_air_quality(raw: &serde_json::Value) -> CurrentAirQuality {
    let c = &raw["current"];
    CurrentAirQuality {
        recorded_at: c["time"].as_str().unwrap().to_string(),
        european_aqi: c["european_aqi"].as_i64().unwrap(),
        us_aqi: c["us_aqi"].as_i64().unwrap(),
        pm10_ugm3: c["pm10"].as_f64().unwrap(),
        pm2_5_ugm3: c["pm2_5"].as_f64().unwrap(),
        carbon_monoxide_ugm3: c["carbon_monoxide"].as_f64().unwrap(),
        nitrogen_dioxide_ugm3: c["nitrogen_dioxide"].as_f64().unwrap(),
        sulphur_dioxide_ugm3: c["sulphur_dioxide"].as_f64().unwrap(),
        ozone_ugm3: c["ozone"].as_f64().unwrap(),
        aerosol_optical_depth: c["aerosol_optical_depth"].as_f64().unwrap(),
        dust_ugm3: c["dust"].as_f64().unwrap(),
        uv_index: c["uv_index"].as_f64().unwrap(),
        uv_index_clear_sky: c["uv_index_clear_sky"].as_f64().unwrap(),
        ammonia_ugm3: c["ammonia"].as_f64().unwrap(),
        alder_pollen_grainsm3: c["alder_pollen"].as_f64().unwrap(),
        birch_pollen_grainsm3: c["birch_pollen"].as_f64().unwrap(),
        mugwort_pollen_grainsm3: c["mugwort_pollen"].as_f64().unwrap(),
        grass_pollen_grainsm3: c["grass_pollen"].as_f64().unwrap(),
        olive_pollen_grainsm3: c["olive_pollen"].as_f64().unwrap(),
        ragweed_pollen_grainsm3: c["ragweed_pollen"].as_f64().unwrap(),
    }
}

pub fn fetch_weather() -> Result<(serde_json::Value, serde_json::Value), Box<dyn std::error::Error>>
{
    let raw = fetch_json(weather_url()?)?;
    let current = serde_json::to_value(parse_current_weather(&raw))?;
    let daily = serde_json::to_value(parse_daily_weather(&raw))?;
    Ok((current, daily))
}

pub fn fetch_air_quality() -> Result<serde_json::Value, Box<dyn std::error::Error>> {
    let raw = fetch_json(air_url()?)?;
    Ok(serde_json::to_value(parse_current_air_quality(&raw))?)
}
