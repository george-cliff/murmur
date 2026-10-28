use rusqlite::Connection;

const PERCH_VERSION_LABEL: &str = "perch-v2";
const PERCH_LABELS_REFERENCE: &str = "perch-v2-labels.csv";

pub fn init_db() -> Result<Connection, Box<dyn std::error::Error>> {
    let conn = rusqlite::Connection::open("data/murmur.db")?;
    conn.execute_batch("PRAGMA foreign_keys = ON;")?;
    let sql = std::fs::read_to_string("schema.sql")?;
    conn.execute_batch(&sql)?;
    Ok(conn)
}

pub fn seed_perch_model_version(conn: &Connection) -> Result<i64, Box<dyn std::error::Error>> {
    conn.execute(
        "INSERT OR IGNORE INTO model_versions (version_label, label_list_reference) VALUES (?1, ?2)",
        rusqlite::params![PERCH_VERSION_LABEL, PERCH_LABELS_REFERENCE],
    )?;

    let id: i64 = conn.query_row(
        "SELECT id FROM model_versions WHERE version_label = ?1",
        rusqlite::params![PERCH_VERSION_LABEL],
        |row| row.get(0),
    )?;

    Ok(id)
}

pub fn get_or_create_model_label(
    conn: &Connection,
    model_version_id: i64,
    class_index: usize,
    raw_label_name: &str,
) -> Result<i64, Box<dyn std::error::Error>> {
    conn.execute(
        "INSERT OR IGNORE INTO model_labels (model_version_id, class_index, raw_label_name, is_taxon)
         VALUES (?1, ?2, ?3, ?4)",
        rusqlite::params![model_version_id, class_index as i64, raw_label_name, true],
    )?;

    let id: i64 = conn.query_row(
        "SELECT id FROM model_labels WHERE model_version_id = ?1 AND class_index = ?2",
        rusqlite::params![model_version_id, class_index as i64],
        |row| row.get(0),
    )?;

    Ok(id)
}
