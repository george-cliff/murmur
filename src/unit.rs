use rusqlite::Connection;

// Publication-safe values for the desktop proof of concept. Replace these with
// configuration before using Murmur for a real deployment.
pub const TEST_UNIT_ID: &str = "desktop-example-unit";
pub const TEST_UNIT_SERIAL: &str = "MM-EXAMPLE-0000";
pub const TEST_LAT: f64 = 0.0;
pub const TEST_LONG: f64 = 0.0;

pub fn register_unit(
    conn: &Connection,
    id: &str,
    serial_number: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    conn.execute(
        "INSERT OR IGNORE INTO units (id, serial_number) VALUES (?1, ?2)",
        rusqlite::params![id, serial_number],
    )?;
    Ok(())
}

pub fn deploy_unit(
    conn: &Connection,
    unit_id: &str,
    started_at: &str,
    latitude: f64,
    longitude: f64,
) -> Result<i64, Box<dyn std::error::Error>> {
    conn.execute(
        "INSERT OR IGNORE INTO deployments (unit_id, started_at, latitude, longitude)
         VALUES (?1, ?2, ?3, ?4)",
        rusqlite::params![unit_id, started_at, latitude, longitude],
    )?;

    let id: i64 = conn.query_row(
        "SELECT id FROM deployments WHERE unit_id = ?1 AND ended_at IS NULL",
        rusqlite::params![unit_id],
        |row| row.get(0),
    )?;

    Ok(id)
}
